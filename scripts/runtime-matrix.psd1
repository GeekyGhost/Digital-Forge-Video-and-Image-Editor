@{
    SchemaVersion = 1

    # The application and LTX share an ABI family, but never a virtual environment.
    Application = @{
        Python = '3.10.11'
        Venv = '.venv'
        Torch = '2.7.1+cu128'
        TorchVision = '0.22.1+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
    }

    Ltx = @{
        Python = '3.10.11'
        Venv = 'vendor\LTX-2\.venv'
        Commit = '4f8905737aac86a554637cac86c178877a39c744'
        LockSha256 = '594E1CCC2A7DC9679D07DC4B1BAB1097CE60FC923909ADB78BAC62ACBDCF211E'
        Torch = '2.7.1+cu128'
        TorchVision = '0.22.1+cu128'
        TorchAudio = '2.7.1+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        SageAttention = '2.2.0'
        SageWheel = 'https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows/sageattention-2.2.0%2Bcu128torch2.7.1-cp310-cp310-win_amd64.whl'
        TritonWindows = '3.3.1.post21'
    }

    # These packs are deliberately isolated because their Python and Torch ABIs differ.
    Moss = @{
        Python = '3.12.10'
        Venv = 'runtimes\moss-sfx'
        Torch = '2.9.0+cu128'
        TorchAudio = '2.9.0+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
    }
    ImageStudio = @{
        Python = '3.13.12'
        Venv = 'vendor\Image-Studio\.venv'
        Torch = '2.10.0+cu128'
        TorchVision = '0.25.0+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        Transformers = '4.57.6'
        DiffusersCommit = '01de02e8b4f2cc91df4f3e91cb6535ebcbeb490c'
    }
    ZImageTrainer = @{
        Python = '3.12.10'
        Venv = 'vendor\Z-Image-Trainer\.venv'
        Torch = '2.11.0+cu128'
        TorchVision = '0.26.0+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        Transformers = '5.17.0'
        DiffusersCommit = '9f1246971270c84dcbe71233edb7a519596a5d02'
    }
    H3 = @{
        Python = '3.12.13'
        Venv = 'vendor\MiniMax-H3\.venv'
        Torch = '2.11.0+cu128'
        TorchVision = '0.26.0+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        Diffusers = '0.40.0'
        Transformers = '4.57.6'
    }
    # VFX Phase 0, task 0.6. Same Python and Torch ABI as the application, so uv hardlinks the
    # wheels it already has; the environment itself is still separate. Revisions pin the weights.
    WanAlpha = @{
        Python = '3.10.11'
        Venv = 'vendor\Wan-Alpha\.venv'
        Commit = '58044108cd4b9896c41d6149f0f2e7195feec166'
        Torch = '2.7.1+cu128'
        TorchVision = '0.22.1+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        Diffusers = '0.40.0'
        Transformers = '5.17.0'
        BaseRevision = 'a064a6c71f5be440641209c07bf2a5ce7a2ff5e4'
        AlphaRevision = '2e99c4e4154db797d18c55d852752ec53c3e1328'
        LightX2VRevision = '8260d429d19fd7a72304cad059160b95d843913f'
    }
    # VFX Phase 4: CoTracker3 (point tracks) and Video Depth Anything Small (depth sequences) share
    # one small environment. Same Python and Torch ABI as the application, so uv hardlinks the wheels.
    # CoTracker3 is CC-BY-NC 4.0 (non-commercial); Video Depth Anything Small is Apache-2.0.
    Motion = @{
        Python = '3.10.11'
        Venv = 'runtimes\motion\.venv'
        Torch = '2.7.1+cu128'
        TorchVision = '0.22.1+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        CoTrackerCommit = '82e02e8029753ad4ef13cf06be7f4fc5facdda4d'
        VideoDepthCommit = '4f5ae23172ba60fd7bc11ef671cca678842c7072'
        CoTrackerCheckpoint = 'facebook/cotracker3:scaled_offline.pth'
        DepthCheckpoint = 'depth-anything/Video-Depth-Anything-Small:video_depth_anything_vits.pth'
    }
    # VFX Phase 5: Wan Animate 2 (distilled 14B) character insertion. Diffusers 0.40.0 ships its
    # modular pipeline; triton-windows compiles FlexAttention (the sparse in-context attention). Same
    # Python and Torch ABI as the application, so uv hardlinks the wheels; the environment is its own.
    # Weights: Apache-2.0. The revision pins them (flattened key layout; no conversion needed).
    WanAnimate = @{
        Python = '3.10.11'
        Venv = 'vendor\Wan-Animate\.venv'
        Torch = '2.7.1+cu128'
        TorchVision = '0.22.1+cu128'
        TorchIndex = 'https://download.pytorch.org/whl/cu128'
        Diffusers = '0.40.0'
        Transformers = '5.17.0'
        TritonWindows = '3.3.1.post21'
        Repo = 'Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers'
        Revision = '36b185201c469c756601cb0779f6597dda1d6c01'
    }
}
