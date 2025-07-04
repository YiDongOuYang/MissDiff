import numpy as np
import torch
import torch.nn as nn
from src.diff_models_table import diff_CSDI
import yaml

import ipdb


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = 0.8  # np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    
    
    def calc_loss_valid( 
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        # In validation, perform T steps forward and backward.
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    # original
    # def calc_loss(
    #     self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    # ):
    #     B, K, L = observed_data.shape
    #     if is_train != 1:  # for validation
    #         t = (torch.ones(B) * set_t).long().to(self.device)
    #     else:
    #         t = torch.randint(0, self.num_steps, [B]).to(self.device)
    #     # ipdb.set_trace()
    #     current_alpha = self.alpha_torch[t]  # (B,1,1)
    #     noise = torch.randn_like(observed_data)
    #     noisy_data = (current_alpha**0.5) * observed_data + (
    #         1.0 - current_alpha
    #     ) ** 0.5 * noise
    #     total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
    #     predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

    #     target_mask = `observed_mask` - cond_mask
    #     residual = (noise - predicted) * target_mask
    #     num_eval = target_mask.sum()
    #     loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
    #     return loss

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        # target_mask = observed_mask - cond_mask
        # residual = (noise - predicted) * target_mask
        residual = (noise - predicted) * observed_mask
        # num_eval = target_mask.sum()
        num_eval = observed_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def generation(self):
        # current_sample = torch.randn_like(observed_data).to(self.device)
        # current_sample = torch.randn(64,1,74).to(self.device)
        # observed_tp = torch.arange(74)
        # current_sample = torch.randn(64,1,111).to(self.device)
        # observed_tp = torch.arange(111)
        current_sample = torch.randn(64,1,9).to(self.device)
        observed_tp = torch.arange(9)
        observed_tp = observed_tp.repeat(64, 1).to(self.device)
        cond_mask = torch.zeros_like(current_sample).to(self.device)
        side_info = self.get_side_info(observed_tp, cond_mask)
        for t in range(self.num_steps - 1, -1, -1):
            if self.is_unconditional == True:
                diff_input = current_sample.unsqueeze(1) # (B,1,K,L) 
            # else:
            #     cond_obs = (cond_mask * observed_data).unsqueeze(1)
            #     noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
            #     diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            predicted = self.diffmodel(
                diff_input, side_info, torch.tensor([t]).to(self.device)
            )  # (B,K,L)
            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

            current_sample = coeff1 * (current_sample - coeff2 * predicted)

            if t > 0:
                noise = torch.randn_like(current_sample)
                sigma = (
                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                current_sample += sigma * noise
        return current_sample.detach()   


    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                # perform T steps forward
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            # perform T steps backward
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )  # (B,K,L)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            # ipdb.set_trace()
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        # In testing, using `gt_mask` (generated with fixed missing rate).
        if is_train == 0:
            cond_mask = gt_mask
        # In training, generate random mask
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        # using for missing ratio=0: testing all kind of missing mechanism
        # missing row: the same as training
        '''
        missing_ratio=0.2
        mask = observed_mask.detach().cpu()

        for col in range(observed_data.shape[0]):  # row #
            obs_indices = np.where(mask[col,: ])[0]
            miss_indices = np.random.choice(
                obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
            )
            gt_mask[col,miss_indices ] = False
        '''
        
        # missing column: different from training
        # for col in range(observed_data.shape[1]):  # col #
        #     obs_indices = np.where(mask[:, col])[0]
        #     miss_indices = np.random.choice(
        #         obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        #     )
        #     gt_mask[miss_indices, col] = False

        with torch.no_grad():
            # original version
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            
            # for fair testing, we should use observed_mask for inference rather than masking more
            
            #version 1: has some problems when there are many entries not observed
            # cond_mask = observed_mask
            # target_mask = torch.ones_like(cond_mask) - cond_mask

            #version 2: also has some problems. It is better to inference only on the data with no missing value
            # cond_mask = gt_mask + (torch.ones_like(observed_mask) - observed_mask)
            # target_mask = observed_mask - gt_mask 

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDIT(CSDI_base):
    def __init__(self, config, device, target_dim=1):
        super(CSDIT, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        # Insert K=1 axis. All mask now with shape (B, 1, L).
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float()

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]

        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
