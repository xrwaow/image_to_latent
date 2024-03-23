import torch
from torch import nn

class resblock(nn.Module):
    def __init__(self, ch):
        super(resblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,  8, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d( 8, 16, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, ch, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
        )
    def forward(self, x): return self.block(x) + x

class print_stuff(nn.Module):
    def __init__(self):
        super(print_stuff, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(  3,  32, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            resblock( 32),

            nn.Conv2d( 32,  64, 3, 2, 1, bias=False),
            nn.LeakyReLU(),
            resblock( 64),

            nn.Conv2d( 64, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(),
            resblock(128),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.LeakyReLU(),
            resblock(256),
        )
    def forward(self, x): return self.block(x)

class to_latent(nn.Module):
    def __init__(self):
        super(to_latent, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(4096, 256, bias=False) for _ in range(8)])
        
    def forward(self, x):
        x = x.reshape(-1,16,4096)
        x = sum(expert(x) for expert in self.experts)
        x = nn.Flatten(-2)(x)
        return x
    
class to_features(nn.Module):
    def __init__(self):
        super(to_features, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(256, 4096, bias=False) for _ in range(8)])

    def forward(self, x):
        x = x.reshape(-1,16,256)
        x = sum(expert(x) for expert in self.experts)
        x = x.reshape(-1,16,64,64)
        return x
    
class to_pixels(nn.Module):
    def __init__(self):
        super(to_pixels, self).__init__()
        ch = 32
        self.block = nn.Sequential(
            nn.Conv2d(16, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            resblock(128),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            resblock(64),
            resblock(64),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            resblock(32),
            resblock(32),

            nn.Conv2d(32, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(),
            resblock(3),
        )

    def forward(self, x): 
        return self.block(x)

class model(nn.Module):
    def __init__(self, debug=False):
        super(model, self).__init__()
        self.fe = feature_extraction()
        self.to_latent = to_latent()
        self.to_features = to_features()
        self.to_pixels = to_pixels()

        self.name = "1.1"
        self.debug = debug
        
    def encode(self, x):
        features = self.fe(x)
        latent = self.to_latent(features)
        return latent
    def decode(self, latent):
        features_decoded = self.to_features(latent)
        pixels = self.to_pixels(features_decoded)
        return pixels
    
    def forward(self, x):
        #encode
        if self.debug: print(f"x shape:                {x.shape}")
        features = self.fe(x)
        if self.debug: print(f"features shape:         {features.shape}")
        latent = self.to_latent(features)
        if self.debug: print(f"latent shape:           {latent.shape}")
        #decode
        features_decoded = self.to_features(latent)
        if self.debug: print(f"features decoded shape: {features_decoded.shape}")
        pixels = self.to_pixels(features_decoded)
        if self.debug: print(f"pixels shape:           {pixels.shape}")

        return pixels