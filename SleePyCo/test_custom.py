import os
import json
import argparse
import warnings
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from models.main_model import MainModel

import matplotlib.pyplot as plt

class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        load_path = os.path.join('checkpoints', self.cfg['name'], 'ckpt_fold-{0:02d}.pth'.format(self.fold))
        model.load_state_dict(torch.load(load_path), strict=False)
        print('[INFO] Model loaded')
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    @torch.no_grad()
    def evaluate(self, input_data):
        inputs = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        self.model.eval()

        outputs = self.model(inputs)
        outputs_sum = torch.zeros_like(outputs[0])
        
        for output in outputs:
            outputs_sum += output

        predicted = torch.argmax(outputs_sum, 1)
        
        print('Predicted: ', predicted)


        
    def run(self):
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        input_data = np.random.rand(1, 1, 30000)
        self.evaluate(input_data)
        print('')

    # def run_csv(self, csv_path):
    #     import pandas as pd
    #     data = pd.read_csv(csv_path, header=0).values[:, 1].squeeze()  # shape: (总点数,)
    #     num_segments = len(data) // 30000
    #     input_data = data[:num_segments*30000].reshape(num_segments, 1, 30000)  # shape: (N, 1, 30000)
    #     self.evaluate(input_data)

    def run_csv(self, csv_path):
        import pandas as pd
        data = pd.read_csv(csv_path, header=0).values[:, 1].squeeze()*2  # shape: (总点数,)

        segment_length = 30000
        frame_length = 3000
        num_segments = len(data) // segment_length

        if num_segments == 0:
            raise ValueError(f"数据长度不足一个完整样本（{segment_length}点）")

        predictions = []

        for i in range(num_segments):
            segment = data[i * segment_length: (i + 1) * segment_length]  # 30000 点
            frames = [segment[j * frame_length: (j + 1) * frame_length] for j in range(10)]  # 拆成 10 帧
            input_frames = np.array(frames)[:, np.newaxis, :]  # shape: (10, 1, 3000)
            inputs = torch.tensor(input_frames, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(inputs)
                output = outputs[0]
                predicted = torch.argmax(output, dim=1)  # shape: (10,)
                predictions.extend(predicted.cpu().numpy())

        df = pd.DataFrame({
            'Frame': range(1, len(predictions) + 1),
            'Predicted': predictions}) 
        filename = 'predictions-ads'
        save_path = r"/home/chenlungan/算法模型/SleePyCo/results/{filename}.csv".format(filename=filename)
        df.to_csv(save_path, index=False)
        print('Predicted per frame:', predictions)

        time_points = [i/120 for i in range(len(predictions))]  # 每30秒一个数据点
        data = predictions
        # 折线图
        fig = plt.gcf()
        fig.clf()
        plt.figure(figsize=(10, 6))

        plt.plot(time_points, data, '#00008B')
        plt.title('Hypnogram Scored by SleePyCo', fontweight="medium")
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time [h]')
        plt.ylabel('Sleep Stage')

        plt.tight_layout()

        filename = 'SC_predictions'
        save_fig_path = r"/home/chenlungan/算法模型/SleePyCo/results/{filename}.png".format(filename=filename)
        plt.savefig(save_fig_path, bbox_inches="tight")
        plt.show()

    def run_npz(self, npz_path):
        data = np.load(npz_path)
        eeg_data = data['x']  # shape: (N, 3000)
        label_data = data['y']  # shape: (N, 1)
        input_data = eeg_data.flatten()  # shape: (N * 3000, )

        segment_length = 30000
        num_segments = len(input_data) // segment_length

        if num_segments == 0:
            raise ValueError(f"数据长度不足一个完整样本（{segment_length}点）")

        predictions = []

        for i in range(num_segments):
            segment = input_data[i * segment_length: (i + 1) * segment_length]  # 30000 点
            frames = [segment[j * 3000: (j + 1) * 3000] for j in range(10)]  # 拆成 10 帧
            input_frames = np.array(frames)[:, np.newaxis, :]  # shape: (10, 1, 3000)
            inputs = torch.tensor(input_frames, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(inputs)
                output = outputs[0]
                predicted = torch.argmax(output, dim=1)  # shape: (10,)
                predictions.extend(predicted.cpu().numpy())

        label_data = label_data[:len(predictions)]

        if len(predictions) != len(label_data):
            print(len(predictions), len(label_data))
            raise ValueError("预测结果长度与标签长度不一致")

        time_points = [i * 1 for i in range(len(predictions))]  # 每30秒一个数据点
        data = predictions
        df = pd.DataFrame({
            'Frame': range(1, len(predictions) + 1),
            'Predicted': predictions,
            'label': label_data}  # 确保标签长度与预测长度一致}
        )
        fig = plt.gcf()
        fig.clf()
        plt.figure(figsize=(10, 6))

        ax_1 = plt.subplot(2, 1, 1)
        ax_1.plot(time_points, label_data[:len(data)], '#008000')

        plt.title('Hypnogram Scored by Human Expert', fontweight="medium")
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time [h]')
        plt.ylabel('Sleep Stage')

        ax_2 = plt.subplot(2, 1, 2)
        ax_2.plot(time_points, data, '#00008B')
        plt.title('Hypnogram Scored by SleePyCo', fontweight="medium")
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time [h]')
        plt.ylabel('Sleep Stage')

        plt.tight_layout()

        filename = 'SC_predictions_npz'
        save_fig_path = r"/home/chenlungan/算法模型/SleePyCo/results/{filename}.png".format(filename=filename)
        plt.savefig(save_fig_path, bbox_inches="tight")
        print('Predicted per frame:', predictions)

    def run_ensemble(self, npz_path):
        from collections import Counter

        # if file_path == "csv_path": # 读取csv文件
        #     data = pd.read_csv(file_path).values
        #     input_data = data[1:, 1].astype(float) # shape: (N, 3000)
        # elif file_path=="npz_path":
        #     data = np.load(file_path)
        #     eeg_data = data['x']
        #     input_data = eeg_data.flatten().astype(float)  # shape N*3000)

        # data = pd.read_csv(csv_path).values
        # input_data = data[1:, 1].astype(float) # shape: (N, 3000)
        #读取npz文件
        data = np.load(npz_path)
        eeg_data = data['x']
        label_data = data['y']
        input_data = eeg_data.flatten().astype(float)  # shape N*3000)

        segment_length = 30000
        num_segments = len(input_data) // segment_length

        if num_segments == 0:
            raise ValueError(f"数据长度不足一个完整样本（{segment_length}点）")

        all_predictions = []  # 存储 20 个模型的预测结果

        for fold in range(1, 11):  # 遍历所有 fold
            print(f"[INFO] 正在加载 Fold {fold} 的模型...")
            self.model = self.build_model()
            load_path = os.path.join('checkpoints', self.cfg['name'], f'ckpt_fold-{fold:02d}.pth')
            self.model.load_state_dict(torch.load(load_path), strict=False)
            self.model.to(self.device)
            self.model.eval()

            fold_preds = []
            fold_total_time = 0.0

            for i in range(num_segments):
                segment = input_data[i * segment_length: (i + 1) * segment_length]  # 30000 点
                frames = [segment[j * 3000: (j + 1) * 3000] for j in range(10)]  # 拆成 10 帧
                input_frames = np.array(frames)[:, np.newaxis, :]  # shape: (10, 1, 3000)
                inputs = torch.tensor(input_frames, dtype=torch.float32).to(self.device)

                start_time = time.time()

                with torch.no_grad():
                    outputs = self.model(inputs)[0]
                    predicted = torch.argmax(outputs, dim=1)  # shape: (10,)
                    fold_preds.extend(predicted.cpu().numpy())
                end_time = time.time()
                fold_total_time += end_time - start_time
            all_predictions.append(fold_preds)
            print(f"[INFO] Fold {fold} 推理耗时: {fold_total_time:.4f}s")
            # 计算总耗时
            total_time = sum(
                (end_time - start_time) for fold_preds in all_predictions
            )

            # 转换为 NumPy 数组：shape (20, T)，T 是总帧数
        all_predictions = np.array(all_predictions)

        # 投票机制：对每一帧选择出现次数最多的类别
        final_prediction = []
        for t in range(all_predictions.shape[1]):  # 对每一帧投票
            frame_votes = all_predictions[:, t]
            most_common = Counter(frame_votes).most_common(1)[0][0]
            final_prediction.append(most_common)

        data = final_prediction
        # 或者更清晰的写法：
        time_points = [i / 120 for i in range(len(data))]  # 每30秒一个数据点（转换为小时）
        
        # # 折线图
        # fig = plt.gcf()
        # fig.clf()
        # plt.figure(figsize=(10, 6))
        #
        #
        # plt.plot(time_points, data, '#008000')
        #
        # plt.title('Hypnogram Scored by SleePyCo', fontweight="medium")
        # plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'R'])
        # plt.xlabel('Time [h]')
        # plt.ylabel('Sleep Stage')
        # plt.show()
        # 折线图
        fig = plt.gcf()
        fig.clf()
        plt.figure(figsize=(10, 6))

        ax_1 = plt.subplot(2, 1, 1)
        ax_1.plot(time_points, label_data[:len(data)], '#008000')

        plt.title('Hypnogram Scored by Human Expert', fontweight="medium")
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time [h]')
        plt.ylabel('Sleep Stage')

        ax_2 = plt.subplot(2, 1, 2)
        ax_2.plot(time_points, data, '#00008B')
        plt.title('Hypnogram Scored by SleePyCo', fontweight="medium")
        plt.yticks([0, 1, 2, 3, 4], ['W', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time [h]')
        plt.ylabel('Sleep Stage')

        plt.tight_layout()

        filename = 'SC_predictions'
        save_fig_path = r"/home/chenlungan/算法模型/SleePyCo/results/{filename}.png".format(filename=filename)
        plt.savefig(save_fig_path, bbox_inches="tight")


        print("最终预测结果（逐帧）：", final_prediction)
        df = pd.DataFrame({
            'Frame': range(1, len(final_prediction) + 1),
            'Predicted': final_prediction,
             'real label': label_data[:len(final_prediction)]  # 假设标签为0，实际使用时请替换为真实标签
        })
        df.to_csv(r"/home/chenlungan/算法模型/SleePyCo/results/SC_predictions.csv", index=False)


        print(f"[INFO] 总推理耗时: {total_time:.4f}s")


        return final_prediction


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--fold', type=int, default=1, help='fold to load checkpoint')
    parser.add_argument('--csv', type=str, default=None, help='path to csv file for evaluation')
    parser.add_argument('--npz', type=str, default=None, help='path to npz file for evaluation')
    parser.add_argument('--ensemble', action='store_true', help='使用 20 折模型集成推理')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility 可复现
    # set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    trainer = OneFoldTrainer(args, args.fold, config)
    # trainer.run()
    if args.ensemble:
        trainer.run_ensemble(args.npz)
    elif args.npz:
        trainer.run_npz(args.npz)
    elif args.csv:
        trainer.run_csv(args.csv)
    else:
        trainer.run()


if __name__ == "__main__":
    main()

