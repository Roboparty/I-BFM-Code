def get_default_fbcpr_config_as_a_dict():
    """Settings from the original fb-cpr example code. This matches ICLR 2025 submission runs"""
    hidden_dim = 1024
    hidden_layers = 2
    nn_model_type = "simple"
    config_dict = {
        "agent": {
            "name": "FBcprAgent",
            "model": {
                "name": "FBcprModel",
                "norm_obs": True,
                "seq_length": 8,
                "actor_std": 0.2,
                "archi": {
                    "f": {"hidden_dim": hidden_dim, "hidden_layers": hidden_layers, "model": nn_model_type},
                    "b": {
                        "norm": True,
                        "hidden_dim": 256,
                        "hidden_layers": 1,
                    },
                    "actor": {
                        "name": "simple",
                        "hidden_dim": hidden_dim,
                        "hidden_layers": hidden_layers,
                    },
                    "critic": {
                        "hidden_dim": hidden_dim,
                        "hidden_layers": hidden_layers,
                    },
                    "discriminator": {
                        "hidden_dim": 1024,
                        "hidden_layers": 3,
                    },
                    "z_dim": 256,
                    "norm_z": True,
                },
            },
            "train": {
                "batch_size": 1024,
                "update_z_every_step": 150,
                "discount": 0.98,
                "grad_penalty_discriminator": 10,
                "weight_decay_discriminator": 0,
                "lr_f": 1e-4,
                "lr_b": 1e-5,
                "lr_actor": 1e-4,
                "lr_critic": 1e-4,
                "lr_discriminator": 1e-5,
                "relabel_ratio": 0.8,
                "use_mix_rollout": True,
                "ortho_coef": 100,
                "train_goal_ratio": 0.2,
                "expert_asm_ratio": 0.6,
                "reg_coeff": 0.01,
                "q_loss_coef": 0.1,
            },
        }
    }
    return config_dict
