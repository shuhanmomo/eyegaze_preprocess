import os
from statistics import mean

from datasets.dataloader360 import EyeGaze360DataLoader
from metircs.suppor_lib import xyz2sphere
from metircs.utils import get_score_filename
import torch
from collections import Iterable, defaultdict
from modules.mdn3d import mixture_probability

class Evaluation:
    def __init__(self, work_dir, writer, val_batch_size, device, seed, max_length,
                 action_map_size, image_input_resize, patch_size, ):

        super(Evaluation, self).__init__()
        self.work_dir = work_dir
        self.writer = writer
        self.val_batch_size = val_batch_size
        self.device = device
        self.max_length = max_length
        self.action_map_size = action_map_size
        self.test_dataiters = {
            "momo":EyeGaze360DataLoader(phase='test',batch_size=val_batch_size,max_length=max_length,seed=seed),
        }

    def validation(self, model, epoch, dataset_name, save=False, ):
        dataset_name = dataset_name.lower()

        model.eval()
        d_model = model.d_model
        save_dir_path = os.path.join(self.work_dir, "seq_results_best_model", dataset_name)

        if save and not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        with torch.no_grad():
            val_performance = defaultdict(list)
            val_data_iter = self.test_dataiters[dataset_name]

            for i_batch, batch in enumerate(val_data_iter):
                val_batch_size = batch['imgs'].size(0)
                enc_masks = None
                # dec_inputs = torch.ones(val_batch_size, 1, 3).to(self.device) * 0.5
                dec_scan = torch.zeros(val_batch_size, 1, 3).to(self.device)
                dec_img = torch.zeros(val_batch_size, 1, 2048).to(self.device)   # if no dynamic features, keep zeros
                dec_sem = torch.zeros(val_batch_size, 1, 512).to(self.device)    # similarly zeros if no semantic info

                imgs = batch['imgs'].to(self.device)

                enc_inputs = model.feature_extrator(imgs)

                if model.replace_encoder:
                    enc_outputs = model.src_emb(enc_inputs)
                    pos_emb = model.pos_emb.repeat(enc_inputs.size(0), 1, 1).to(enc_outputs.device)
                    enc_outputs = enc_outputs + pos_emb
                else:
                    enc_outputs, enc_self_attns = model.encoder(enc_inputs, None)
                for n in range(self.max_length):
                    fused = model.fixation_embed(dec_scan, dec_img, dec_sem)
                    cls_token_expanded = model.cls_token.expand(val_batch_size, -1, -1)
                    fused_with_cls = torch.cat([cls_token_expanded, fused], dim=1)  
                    dec_masks = torch.zeros(val_batch_size, fused_with_cls.size(1)).to(self.device)
                    dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(enc_outputs, fused_with_cls,
                                                                               enc_masks=enc_masks,
                                                                               dec_masks=dec_masks.to(
                                                                                   self.device))

                    pis, mus, sigmas, rhos = model.mdn(dec_outputs[:,1:,:])
                    outputs = model.mdn.sample_prob(pis, mus, sigmas, rhos).to(self.device)
                    last_fixations = outputs[:, -1]

                    dec_scan = torch.cat([dec_scan, last_fixations.unsqueeze(1)], dim=1)
                    zero_img = torch.zeros(val_batch_size, 1, 2048).to(self.device)
                    zero_sem = torch.zeros(val_batch_size, 1, 512).to(self.device)
                    dec_img = torch.cat([dec_img, zero_img], dim=1)
                    dec_sem = torch.cat([dec_sem, zero_sem], dim=1)
                probs = mixture_probability(pis, mus, sigmas, rhos,
                                            batch['scanpath'].unsqueeze(-1).to(self.device)).squeeze()

                # 筛选有效注视点预测概率
                probs_mask = torch.arange(self.max_length).expand(val_batch_size, self.max_length). \
                    lt(batch['valid_len'].unsqueeze(-1).expand(val_batch_size, self.max_length)).to(self.device)

                probs = torch.masked_select(probs, probs_mask)
                loss = torch.mean(-torch.log(probs))
                loss_item = loss.detach().cpu().item()

                outputs = dec_scan[:, 1:, :].cpu().numpy()  # 用解码器输入作为预测点

                for batch_index in range(val_batch_size):
                    output = outputs[batch_index]
                    output_sphere = xyz2sphere(torch.from_numpy(output))
                    print(dataset_name, "--", batch['file_names'][batch_index])
                    if save:
                        save_path = os.path.join(save_dir_path, batch['file_names'][batch_index] + '.pck')
                        torch.save(output_sphere, save_path)
                    else:
                        # scores = get_score_filename(pred_fixation_sphere=output_sphere.numpy(),
                        #                             file_name=batch['file_names'][batch_index],
                        #                             dataset_name=dataset_name, )

                        val_performance['loss'].append(loss_item)
                        # for metric, score in scores.items():
                        #     val_performance[metric].append(score)

                        print(f"Epoch: {epoch}, val_loss = {loss_item}")
                if not save and i_batch > 5:
                    break
            for metric, scores_list in val_performance.items():
                val_performance[metric] = mean(scores_list)

            if self.writer and not save:
                # tensorboard
                print('epoch', epoch, f'{dataset_name}_val_performance', val_performance, )
                for key, value in val_performance.items():
                    print(key, value, isinstance(value, Iterable))
                    if isinstance(value, Iterable):
                        for index, sub_value in enumerate(value):
                            self.writer.add_scalar(f'val_{dataset_name}_{key}/score{index}', sub_value, epoch)
                    elif key == "loss":
                        self.writer.add_scalar(f'AA_Scalar/val_{dataset_name}_{key}', value, epoch)
                    else:
                        self.writer.add_scalar(f'val_{dataset_name}/{key}', value, epoch)
        return val_performance['loss']
