import configparser


def read_or_default(config, section, option, field_type, default=None):
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
        num_convs(int): Number of convolutions per Interlacer layer
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

        self.dataset = read_or_default(config, 'DATA', 'dataset', 'str')
        self.task = read_or_default(
            config, 'DATA', 'task', 'str')
        self.us_frac = read_or_default(
            config, 'DATA', 'us_frac', 'float')
        self.mot_frac = read_or_default(
            config, 'DATA', 'mot_frac', 'float')
        self.max_htrans = read_or_default(
            config, 'DATA', 'max_htrans', 'float')
        self.max_vtrans = read_or_default(
            config, 'DATA', 'max_vtrans', 'float')
        self.max_rot = read_or_default(
            config, 'DATA', 'max_rot', 'float')
        self.noise_std = read_or_default(
            config, 'DATA', 'noise_std', 'float')

        self.architecture = read_or_default(
            config, 'MODEL', 'architecture', 'str')
        self.kernel_size = read_or_default(
            config, 'MODEL', 'kernel_size', 'int')
        self.num_features = read_or_default(
            config, 'MODEL', 'num_features', 'int')
        self.num_convs = read_or_default(
            config, 'MODEL', 'num_convs', 'int', 1)
        self.num_layers = read_or_default(
            config, 'MODEL', 'num_layers', 'int')
        self.loss_type = read_or_default(
            config, 'MODEL', 'loss_type', 'str')
        self.loss = read_or_default(config, 'MODEL', 'loss', 'str')
        self.loss_lambda = read_or_default(
            config, 'MODEL', 'loss_lambda', 'float')
        self.input_domain = read_or_default(
            config, 'MODEL', 'input_domain', 'str')
        self.output_domain = read_or_default(
            config, 'MODEL', 'output_domain', 'str')
        self.nonlinearity = read_or_default(
            config, 'MODEL', 'nonlinearity', 'str')
        self.enforce_dc = read_or_default(
            config, 'MODEL', 'enforce_dc', 'bool')

        self.num_epochs = read_or_default(
            config, 'TRAINING', 'num_epochs', 'int')
        self.batch_size = read_or_default(
            config, 'TRAINING', 'batch_size', 'int')
        self.set_job_name()

    def set_job_name(self):
        """Set job name for storing training logs."""
        self.job_name = self.dataset
        for tag in [
                self.task,
                self.us_frac,
                self.mot_frac,
                self.max_htrans,
                self.max_vtrans,
                self.max_rot,
                self.noise_std,
                self.architecture,
                self.kernel_size,
                self.num_features,
                self.num_convs,
                self.num_layers,
                self.loss_type,
                self.loss,
                self.loss_lambda,
                self.input_domain,
                self.output_domain,
                self.nonlinearity,
                self.enforce_dc,
                self.num_epochs,
                self.batch_size]:
            self.job_name += '-' + str(tag)
        self.job_name += ''
