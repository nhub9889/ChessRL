import torch
import argparse
from src.pipelines import TrainingPipeline, Visualizer
from src.model import Model

parser = argparse.ArgumentParser(description= 'Model chess base on AlphaGo')
parser.add_argument('--train', type= bool, default= True, dest= 'flag')
parser.add_argument('--pgn_path', type = str, default= "data")
parser.add_argument('--model_path', type = str, default= "checkpoints")
parser.add_argument('--device', type = str, default= 'cuda')
parser.add_argument('--output', type = str, default = 'model')
def main():
    args = parser.parse_args()
    device = torch.device(args.device)
    model = Model(input_channels= 18, actions=64*64, device= device)

    if args.flag:
        path = args.pgn_path
        output_path = args.output
        visualizer = Visualizer()
        pipeline = TrainingPipeline(model, visualizer= visualizer)
        pipeline.supervised_train(path)

        pipeline.train()

        model.save(output_path)
        visualizer.generate_report()
        visualizer.save_plots()


