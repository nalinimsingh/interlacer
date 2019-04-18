import configparser


def read_or_default(config, section, option, field_type, default):
    """Reads and returns a config field.

    Args:
      config(ConfigParser): ConfigParser object with loaded config data
      section(str): Section in INI file of field to be loaded
      option(str): Option in INI file of field to be loaded
      field_type(str): Field to be loaded (TODO: make actual types)
      default(field_type): Default value for field

    Returns:
      Value of field, or if not specified, default value

    """
    try:
        if(field_type == 'int'):
            return config.getint(section, option)
        elif(field_type == 'float'):
            return config.getfloat(section, option)
        elif(field_type == 'str'):
            return config.get(section, option)
        elif(field_type == 'bool'):
            return config.getboolean(section, option)
    except BaseException:
        return default


class TrainingConfig:
    """Stores data, model, and optimization settings for training.

    Attributes:
        dataset(str): Data modality to train on ('MRI' or 'MNIST')
        task(str): Type of data corruption to apply ('undersample', 'motion', or 'noise')
        corruption_frac(float): Parameter specifying corruption severity

        architecture: Model architecture ('CONV','CONV_RESIDUAL',or 'INTERLACER_RESIDUAL')
        kernel_size(int): Size of kernel in intermediate layers
        num_features(int): Number of features in intermediate layers
        num_layers(int): Number of layers in model
        loss_type(str): Domain in which to compute loss ('IMAGE' or 'FREQ')
        loss(str): Loss function ('L1' or 'L2')
        input_domain(bool): Domain of network input ('IMAGE' or 'FREQ')
        output_domain(bool): Domain of network output ('IMAGE' or 'FREQ')
        nonlinearity(str): Nonlinearity to follow layers ('relu' or '3-piece')

        num_epochs(int): Number of training epochs
        batch_size(int): Batch size

    """

    def __init__(self, config_path):
        self.config_path = config_path

    def read_config(self):
        """Read in fields from INI config file."""
        config = configparser.ConfigParser()
        config.read(self.config_path)

        self.dataset = read_or_default(config, 'DATA', 'dataset', 'str', 'MRI')
        self.task = read_or_default(
            config, 'DATA', 'task', 'str', 'undersample')
        self.corruption_frac = read_or_default(
            config, 'DATA', 'corruption_frac', 'float', 0.50)

        self.architecture = read_or_default(
            config, 'MODEL', 'architecture', 'str', 'CONV')
        self.kernel_size = read_or_default(
            config, 'MODEL', 'kernel_size', 'int', 9)
        self.num_features = read_or_default(
            config, 'MODEL', 'num_features', 'int', 32)
        self.num_layers = read_or_default(
            config, 'MODEL', 'num_layers', 'int', 6)
        self.loss_type = read_or_default(
            config, 'MODEL', 'loss_type', 'str', 'image')
        self.loss = read_or_default(config, 'MODEL', 'loss', 'str', 'L1')
        self.input_domain = read_or_default(
            config, 'MODEL', 'input_domain', 'str', 'FREQ')
        self.output_domain = read_or_default(
            config, 'MODEL', 'output_domain', 'str', 'FREQ')
        self.nonlinearity = read_or_default(
            config, 'MODEL', 'nonlinearity', 'str', 'relu')

        self.num_epochs = read_or_default(
            config, 'TRAINING', 'num_epochs', 'int', 500)
        self.batch_size = read_or_default(
            config, 'TRAINING', 'batch_size', 'int', 10)
        self.set_job_name()

    def set_job_name(self):
        """Set job name for storing training logs."""
        self.job_name = self.dataset
        for tag in [
                self.task,
                self.corruption_frac,
                self.architecture,
                self.kernel_size,
                self.num_features,
                self.num_layers,
                self.loss_type,
                self.loss,
                self.input_domain,
                self.output_domain,
                self.nonlinearity,
                self.num_epochs,
                self.batch_size]:
            self.job_name += '-' + str(tag)
        self.job_name += ''
