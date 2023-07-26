import os.path
from stable_baselines3 import PPO
import torch as th
import argparse


class ExportablePolicyCPU(th.nn.Module):
    def __init__(self, extractor1, extractor2, action_net):
        super().__init__()
        self.extractor1 = extractor1.cpu()
        self.extractor2 = extractor2.cpu()
        self.action_net = action_net.cpu()

    def forward(self, observation):
        return self.action_net(self.extractor2(self.extractor1(observation.cpu()))[0])

class ExportablePolicyCUDA(th.nn.Module):
    def __init__(self, extractor1, extractor2, action_net):
        super().__init__()
        self.extractor1 = extractor1.cuda()
        self.extractor2 = extractor2.cuda()
        self.action_net = action_net.cuda()

    def forward(self, observation):
        return self.action_net(self.extractor2(self.extractor1(observation.cuda()))[0])



def export_policy(input_filename: str, output_filename: str, export_type: str, cuda: bool) -> None:
    if not os.path.exists(input_filename):
        print(f"Error. File {input_filename} does not exist")
        exit(-1)
    model = PPO.load(input_filename)
    if cuda:
        print("Exporting for CUDA")
        exportable = ExportablePolicyCPU(model.policy.pi_features_extractor, model.policy.mlp_extractor,
                                  model.policy.action_net)
    else:
        print("Exporting for CPU")
        exportable = ExportablePolicyCUDA(model.policy.pi_features_extractor, model.policy.mlp_extractor,
                                  model.policy.action_net)


    observation_size = model.policy.observation_space.shape
    dummy_input = th.ones(1, *observation_size)

    if export_type.lower() == 'onnx':
        th.onnx.export(
            exportable,
            dummy_input,
            output_filename,
            opset_version=9,  # TODO: check which version is best here
            input_names=["input"]
        )
    else:
        sm = th.jit.script(exportable)
        sm.save(output_filename)

    print("Policy exported successfully!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Policy export utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Input file. A .zip archive with PPO model", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument("-t", "--export-type", default="torch", help="Type of exported policy. Either 'torch' for "
                                                                     "pyTorch .pt file or 'ONNX' for ONNX file format.")
    parser.add_argument("-d", "--device", default="cpu", help="Type of device to run policy at. Either 'cpu' or 'cuda'")

    args = parser.parse_args()
    config = vars(args)

    if config['export_type'].lower() not in ('torch', 'onnx'):
        print(f"Not supported export type: {config['export_type']}. Should be 'torch' or 'ONNX'")
        exit(-1)

    export_policy(config['input'], config['output'], config['export_type'], config['device'] == 'cuda')
