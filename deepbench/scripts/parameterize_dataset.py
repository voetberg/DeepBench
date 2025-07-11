
import inspect
import os
import h5py
import numpy as np
import yaml
import uuid

class ParameterizeDataset:
    def __init__(self, config): 
        self.config_path = config
        self.config = yaml.safe_load(open(config))
        self.engine = self.initialize_data_engine()

        seed = self.config.get("seed", None)
        if seed is None:
            seed = np.random.randint(0, 10000)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.run_id = self.config.get("run_id", uuid.uuid4().hex)

        self.theta_params = self.generate_parameters().keys()
        self.x_params = self.generate_object_parameters().keys()
        self.y_shape = self.engine(**self.generate_parameters()).create_object(**self.generate_object_parameters()).shape  # Assuming create_object returns a numpy array

    def initialize_data_engine(self): 
        type_ = self.config.get("object_type")
        name = self.config.get("object_name")

        if type_ == "astro":
            import deepbench.astro_object as module
        elif type_ == "shape":
            import deepbench.shapes as module
        elif type_ == "physics": 
            import deepbench.physics_object as module
        else: 
            raise ValueError(f"Unsupported object type: {type_}")
        
        try: 
            obj = getattr(
                module, name
            )

        except AttributeError:
            raise ValueError(f"Object {name} not found in module {module.__name__}")

        return obj

    def generate_parameters(self) -> dict: 
        "Create a dictionary of N random parameters for each object in image_parameters"
        given_params = self.config.get("image_parameters", {})
        generated_parameters = {
            key: self.rng.uniform(
                low=param_range[0],
                high=param_range[1],
            ) for key, param_range in given_params.items()
        }
        return generated_parameters

    def generate_object_parameters(self) -> dict: 
        args = self.config.get("object_parameters", {})
        if args is None:
            args = inspect.signature(self.engine.create_object).parameters
        
        return {
            key: self.rng.uniform(
                low=param_range[0],
                high=param_range[1]
            ) for key, param_range in args.items()
        }

    def make_dataset(self): 
        n_samples = self.config.get("total_runs", 100)
        parameter_noise = self.config.get("parameter_noise", 0.01)

        thetas = np.zeros((n_samples, len(self.theta_params)))
        xs = np.zeros((n_samples, len(self.x_params)))
        ys = np.zeros((n_samples, *self.y_shape))

        for index in range(n_samples):
            theta = self.generate_parameters()
            input_theta = theta.copy()
            if "parameter_noise" in inspect.signature(self.engine).parameters:
                input_theta['parameter_noise'] = parameter_noise

            if "noise" in inspect.signature(self.engine).parameters:
                input_theta['noise'] = parameter_noise
            
            else: 
                input_theta = {
                    key: value * self.rng.uniform(1 - parameter_noise, 1 + parameter_noise)
                    for key, value in theta.items()
                }

            engine = self.engine(
                **input_theta, 
            )
            x = self.generate_object_parameters()
            y = engine.create_object(**x)

            thetas[index] = np.array(list(theta.values()))
            xs[index] = np.array(list(x.values()))
            ys[index] = y

        return {
            "thetas": thetas,
            "xs": xs,
            "ys": ys
        }

    def save_dataset(self, dataset, name):

        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))

        if os.path.exists(name):
            file_name = f"{self.run_id}_{os.path.basename(name)}"
            name = os.path.join(os.path.dirname(name), file_name)

        if not name.endswith(".h5"):
            name += ".h5"

        with h5py.File(name, "w") as f:
            f.create_dataset("thetas", data=dataset["thetas"])
            f.create_dataset("xs", data=dataset["xs"])
            f.create_dataset("ys", data=dataset["ys"])

            # Add additional metadata
            f.attrs['config'] = self.config_path
            f.attrs['seed'] = self.seed
            f.attrs['run_id'] = self.run_id
            f.attrs['theta_columns'] = list(self.theta_params)
            f.attrs['x_columns'] = list(self.x_params)

    def __call__(self, name):
        """Run the dataset generation process."""
        dataset = self.make_dataset()
        self.save_dataset(dataset, name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--name", type=str, required=True, help="Path to output file. Directory will be created if it does not exist.")

    args = parser.parse_args()
    ParameterizeDataset(args.config)(args.name)