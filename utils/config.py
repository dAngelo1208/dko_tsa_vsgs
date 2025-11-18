import argparse
import random

# TODO: 1) Deeper and wider encoder and decoder
#       2) Training with pre-normalized data.       B-ETTER
#       3）Training with wider koopman dimision
#       4) Training with more complex pairs         BETTER: MORE LINEAR KOOPMAN
#       5) Training with time[0] reconstruction loss
#       3) Training with real power system data

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    
    # train_name = 'nonlinear_data200p_dim2/nonlineardata-ShallowBaseline-24kdim-88egin-010110000-newlossVarConv0003-gradclip5'
    train_name = 'VSGPaperPlot/Refined-3dyn_25-15s_6081_tanh_samr_1e-2'
    disable_log = False  # Set to True to disable logging with wandb and swanlab
    
    # training technique parameters
    # data_name = 'VSG_2dynX_symSample4-4-44_p306_8k_nonorm_100p'
    data_name = 'VSG_3dyn_6081_35151_200p'
    delta_t = 0.02
    len_time = 100
    learning_rate = 1e-3
    omeganet_lr_ratio = 1
    weight_decay = 0
    weight_decay_omega = 0
    scheduler = True
    scheduler_epochs = 13000
    act_func = 'tanh'
    grad_clip = True
    warmup_steps = 800
    encoder_first = 200
    relative_loss = False
    
    # loss parameters
    recon_lam = 0.1
    pred_lam = 0.1
    koopman_lam = 1.0
    egin_lam = 0.01
    sea_lam = 0.0
    sea_r_start = 1.1
    var_lam = 0.0
    cov_lam = 0.0
    Linf_lam = 1e-3
    sdml_tau = 0.2
    sdml_mu_margin = 0.0

    # model parameters
    origin_dim = 3
    # shifts_for_windows = len_time - 1
    # shifts_for_prediction = len_time - 1
    # koopman_sihfts = len_time - 1
    shifts_for_windows = 59
    shifts_for_prediction = 59
    koopman_sihfts = 59
    koopman_dim = 36
    num_complex_pairs = 12
    num_real = 12
    num_splits = 1
    
    residual = False
    encoder_layer = [origin_dim, 128, 128, 128, koopman_dim]
    decoder_layer = [koopman_dim, 128, 128, 128, origin_dim]
    # encoder_layer = [origin_dim, 64, 128, 128, 64, koopman_dim]
    # decoder_layer = [koopman_dim, 64, 128, 128, 64, origin_dim]
    enc_film_idx = [0, 1, 2]  # Indices of layers to apply FiLM
    omg_film_idx = [0, 1]  # Indices of layers to apply FiLM for omega
    p_dim = 6  # Dimension of the parameter vector, e.g., for VSG parameters
    emb_dim = 64  # Embedding dimension for FiLM layers
    encoder_layer_res = [origin_dim, [256, 256], [512, 512], [256, 256], koopman_dim]
    decoder_layer_res = [koopman_dim, [256, 256], [512, 512], [256, 256], origin_dim]
    
    # training settings
    # resume = r'logs\CASE00-newdata-dim7-50sample200p\ShallowBaseline-24kdim-88egin-010110004-newlossVarConv0003-gradclip5\checkpoints\checkpoint_step_5995.pth'
    resume = False
    run_id = 'qhd9kges'
    batch_size = 18000
    epochs = 13000
    cuda_device = 'cuda:1'
    multi_gpu = False
    random_seed = True
    # random_seed = 34  # Set a fixed seed for reproducibility
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="DeepKoopman Training Configuration")

    # Directories
    parser.add_argument('--run_id', type=str, default=run_id, help='Run ID for tracking')
    parser.add_argument('--data_dir', type=str, default='data_fnl_exp', help='Path to the dataset')
    parser.add_argument('--data_name', type=str, default=data_name, help='Name of the dataset')
    parser.add_argument('--data_train_len', type=int, default=1, help='Number of training files')
    parser.add_argument('--log_dir', type=str, default='logs'+'/'+train_name, help='Directory for logs')
    parser.add_argument('--tb_dir', type=str, default='tb_logs'+'/'+train_name, help='Directory for TensorBoard logs')
    parser.add_argument('--train_name', type=str, default=train_name, help='Name of the training')
    parser.add_argument('--origin_dim', type=int, default=origin_dim, help='Dimension of the original data')
    parser.add_argument('--delta_t', type=float, default=delta_t, help='Time step delta t')
    
    # Data Process
    parser.add_argument('--len_time', type=int, default=len_time, help='Length of time series')
    parser.add_argument('--num_shifts_max', type=int, default=None, help='Number of shifts')
    parser.add_argument('--num_splits', type=int, default=num_splits, help='Number of splits')
    
    # Time shifts
    parser.add_argument('--shifts', type=int, default=koopman_sihfts, help='Shifts for the Koopman operator')
    parser.add_argument('--shifts_input', type=int, default=shifts_for_windows, help='Shifts for input')
    parser.add_argument('--shifts_pred', type=int, default=shifts_for_prediction, help='Shifts for prediction')
    
    # Koopman settings
    parser.add_argument('--koopman_dim', type=int, default=koopman_dim, help='Koopman space dimension')
    parser.add_argument('--num_real', type=int, default=num_real, help='Number of real eigenvalues')
    parser.add_argument('--num_complex_pairs', type=int, default=num_complex_pairs, help='Number of complex conjugate pairs')
    parser.add_argument('--widths_omega_complex', type=int, nargs='+', default=[1, 32, 32, 2], help='Widths of omega complex MLP layers')
    parser.add_argument('--widths_omega_real', type=int, nargs='+', default=[1, 32, 32, 1], help='Widths of omega real MLP layers')

    # Model hyperparameters
    parser.add_argument('--residual', type=bool, default=residual, help='Residual connection')
    parser.add_argument('--act_type', type=str, default=act_func, help='Activation function type')
    parser.add_argument('--encoder_layer', type=list, default=encoder_layer_res if residual else encoder_layer, help='Hidden dimensions of encoder')
    parser.add_argument('--decoder_layer', type=list, default=decoder_layer_res if residual else decoder_layer, help='Hidden dimensions of decoder')
    parser.add_argument('--enc_film_idx', type=tuple, default=enc_film_idx, help='Indices of layers to apply FiLM')
    parser.add_argument('--omg_film_idx', type=tuple, default=omg_film_idx, help='Indices of layers to apply FiLM for omega')
    parser.add_argument('--emb_dim', type=int, default=emb_dim, help='Embedding dimension for FiLM layers')
    parser.add_argument('--p_dim', type=int, default=p_dim, help='Dimension of the parameter vector')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in encoder/decoder')
    parser.add_argument('--batchnorm', type=str2bool, default=False, help='Batch normalization')
    
    # Loss hyperparameters
    parser.add_argument('--relative_loss', type=bool, default=relative_loss, help='Relative loss')
    parser.add_argument('--recon_lam', type=float, default=recon_lam, help='Reconstruction loss weight')
    parser.add_argument('--pred_lam', type=float, default=pred_lam, help='Prediction loss weight')
    parser.add_argument('--koopman_lam', type=float, default=koopman_lam, help='Koopman loss weight')
    parser.add_argument('--sea_lam', type=float, default=sea_lam, help='SEA loss weight')
    parser.add_argument('--sea_r_start', type=float, default=sea_r_start, help='Initial relaxed radius for SEA')
    parser.add_argument('--sdml_tau', type=float, default=sdml_tau, help='Temperature parameter for SDML')
    parser.add_argument('--sdml_mu_margin', type=float, default=sdml_mu_margin, help='Real-part margin for SDML')
    parser.add_argument('--denominator_nonzero', type=float, default=1e-5, help='Denominator for relative loss')
    parser.add_argument('--Linf_lam', type=float, default=Linf_lam, help='Linf loss weight')
    parser.add_argument('--L1_lam', type=float, default=1e-10, help='L1 loss weight')
    parser.add_argument('--L2_lam', type=float, default=1e-10, help='L2 loss weight')
    parser.add_argument('--egin_lam', type=float, default=egin_lam, help='Eigenvalue loss weight')
    parser.add_argument('--var_lam', type=float, default=var_lam, help='Variance loss weight')
    parser.add_argument('--cov_lam', type=float, default=cov_lam, help='Covariance loss weight')
    parser.add_argument('--encoder_first', type=int, default=encoder_first, help='Encoder first epochs')
    parser.add_argument('--all_step_recong_loss', type=int, default=5, help='All step reconstruction loss epochs')

    # Training hyperparameters
    parser.add_argument('--resume', type=str, default=resume, help='Path to the checkpoint')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer type adam/sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--warm_up', type=int, default=warmup_steps, help='Warm up steps.')
    parser.add_argument("--weight_decay", type=float, default=weight_decay, help="Weight decay for optimizer")
    parser.add_argument('--weight_decay_omega', type=float, default=weight_decay_omega, help='Weight decay for omega networks')
    parser.add_argument('--warm_up_lr', type=float, default=1e-5, help='Warm up learning rate')
    parser.add_argument('--scheduler', type=bool, default=scheduler, help='Scheduler True or False')
    parser.add_argument('--scheduler_epochs', type=int, default=scheduler_epochs, help='Scheduler epochs')
    parser.add_argument('--min_lr', type=float, default=2e-5, help='Minimum learning rate')
    parser.add_argument('--omeganet_lr_ratio', type=float, default=omeganet_lr_ratio, help='OmegaNet learning rate ratio')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=epochs, help='Number of training epochs')
    parser.add_argument('--eval_iter', type=int, default=5, help='Number of epochs between evaluations')
    parser.add_argument('--plot_iter', type=int, default=30, help='Number of epochs between model plot distributions')
    parser.add_argument('--gradient_clip', type=bool, default=grad_clip, help='Gradient clip')
    parser.add_argument('--disable_log', type=str2bool, default=disable_log, help='Disable wandb & swanlab logging (True/False).')

    # SEED
    parser.add_argument('--seed', type=int, default=random.randint(0, 200) if random_seed is True else random_seed, help='Random seed')
    
    # Device
    import torch
    device = cuda_device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=device, help='GPU index (e.g., "cuda:0" or "cuda:1")')
    parser.add_argument('--multi_gpu', action='store_true', default=multi_gpu, help='Enable multi GPU training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Pin memory for data loading')
    
    args, unknown = parser.parse_known_args()

    return args

if __name__ == "__main__":
    config = get_config()
    print(f"Activation function type: {config.act_type}")
    print(f"Input dimension: {config.input_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Koopman space dimension: {config.koopman_dim}")
    print(f"Number of layers: {config.num_layers}")
    print(f"Number of complex conjugate pairs: {config.num_complex_pairs}")
    print(f"Number of real eigenvalues: {config.num_real}")
    print(f"Time step delta t: {config.delta_t}")
    print(f"Shifts: {config.shifts}")
    print(f"Widths of omega complex MLP layers: {config.widths_omega_complex}")
    print(f"Widths of omega real MLP layers: {config.widths_omega_real}")
