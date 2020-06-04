Software: Minimum prerequisites:
Dataset from the UAV-II-Dataset processing routines. We expect three directories which contains different parts of the dataset:
1) A path to the TRAINING data from the Indoor-NYU dataset 
2) A path to the EVALUATION data from the Indoor-NYU dataset
NOTE HERE: Your need to specify such a path and you need to have the same directory structure as in the other parts. But the evaluation part is the LEAST interesting one because a quantitative evaluation is of little interest up to this point of model development.
3.) A path to the TRAINING data from your own data acquisition

Hardware:
Consider a single Geforce RTX 2080 GPU or GTX 1080 with 8GB RAM as the absoulute minimum for trainng. With a 1080 TI and 12 GB of memory you can increase the Batch size to 8. 

parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=655.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=65536.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--own_data_dir', dest='own_data_dir', action='store_true', help='Specify the path to the directory which contains the training dataset from your own data acquisition with the Microsoft Kinect v1.')
parser.add_argument('--nyu_train_data_dir', dest='nyu_train_data_dir', action='store_true', help='Specify the path to the directory which contains the training dataset which was generated from the NYU indoor dataset.')
parser.add_argument('--nyu_eval_data_dir', dest='nyu_eval_data_dir', action='store_true', help='Specify the path to the directory which contains the evaluation dataset which was generated from the NYU indoor dataset.')
