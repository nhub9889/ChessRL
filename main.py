import torch
import argparse
from src.pipelines import TrainingPipeline
from src.model import Model
from src.visualizer import Visualizer

parser = argparse.ArgumentParser(description= 'Model chess base on AlphaGo')
parser.add_argument('--train', type= bool, default= True, dest= 'flag')
parser.add_argument('--pgn_path', type = str, default= "data")
parser.add_argument('--model_path', type = str, default= "checkpoints")
parser.add_argument('--device', type = str, default= 'cuda')
parser.add_argument('--output', type = str, default = 'model')
parser.add_argument('--batch_size', type = int, default= 128)
parser.add_argument('--iterations', type = int, default= 2048)
parser.add_argument('--supervised_epochs', type = int, default= 20)
parser.add_argument('--supervised_batch_size', type = int, default= 64)
parser.add_argument('--num_workers', type= int, default= 4)
parser.add_argument('--simulations_start', type= int, default= 100)
parser.add_argument('--simulations_end', type= int, default= 300)
parser.add_argument('--max_moves_start', type= int, default= 100)
parser.add_argument('--max_moves_end', type= int, default= 300)
parser.add_argument('--schedule_mode', type= str, default= 'decrease')
parser.add_argument('--schedule_type', type= str, default= 'linear')
parser.add_argument('--schedule_step', type= int, default= 800)
def main():
    args = parser.parse_args()
    device = torch.device(args.device)
    model = Model(input_channels= 18, actions=64*64, device= device)

    if args.flag:
        path = args.pgn_path
        output_path = args.output
        visualizer = Visualizer()
        pipeline = TrainingPipeline(model, batch_size= args.batch_size,
                                    supervised_batch_size= args.supervised_batch_size, num_workers= args.num_workers,
                                    supervised_epochs= args.supervised_epochs, iterations= args.iterations,
                                    visualizer= visualizer, simulations_start= args.simulations_start,
                                    simulations_end= args.simulations_end,
                                    max_moves_start= args.max_moves_start, max_moves_end= args.max_moves_end,
                                    schedule_mode= args.schedule_mode, schedule_type= args.schedule_type,
                                    schedule_steps= args.schedule_step)
        # pipeline.supervised_train(path)

        pipeline.train()

        model.save(output_path)
        visualizer.generate_report()
        visualizer.save_plots()

if __name__ == '__main__':
    main()

