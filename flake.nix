{
  description = "torchvision-fasterrcnn";

  nixConfig = {
    substituters = [
      https://cache.nixos.org
    ];
    bash-prompt = "\[nix-develop\]$ ";
  };
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
    hasktorch-datasets.url = "github:hasktorch/hasktorch-datasets";
    # poetry2nix = {
    #   url = "github:nix-community/poetry2nix?rev=046bddd753f99ce73ab2fbb91a2d5f620eadae39";
    #   flake = false;
    # };
  };

  outputs = { self, nixpkgs, flake-utils, hasktorch-datasets }: #, poetry2nix
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };
        customOverrides = self: super: {
          # Overrides go here
        };

        packageName = "torchvision-fasterrcnn";
        bdd100k = hasktorch-datasets.packages.${system}.datasets-bdd100k-coco;
        bdd100k-mini = hasktorch-datasets.packages.${system}.datasets-bdd100k-mini-coco;
        sample-images = hasktorch-datasets.packages.${system}.datasets-sample-images;
        img2coco = hasktorch-datasets.lib.${system}.datasets.img2coco;
        hasktorch-datasets-utils = hasktorch-datasets.lib.${system}.utils;
        src2drv = hasktorch-datasets.lib.${system}.datasets.src2drv;
        fasterrcnn = pkgs.callPackage ./nix/default.nix {
          inherit bdd100k;
          inherit bdd100k-mini;
          inherit hasktorch-datasets-utils;
          inherit src2drv;
        };
        sample-image = src2drv {
          srcs = [
            (builtins.fetchurl {
              name = "b1d0091f-f2c2d2ae.zip";
              url = "https://github.com/junjihashimoto/torchvision-fasterrcnn/releases/download/dataset/b1d0091f-f2c2d2ae.zip";
              sha256 = "0mz0dv1vvqffyk863mzxnvpsiklmh4ixhxr1yfl5017fm8miln2f";
            })
          ];
        };
      in {
        lib = {
          train = args@{...}: fasterrcnn.train args;
          test = args@{...}: fasterrcnn.test args;
          detect = args@{...}: fasterrcnn.detect args;
          finetuning = args@{...}: fasterrcnn.finetuning args;
          gen-feature-map-with-partial-image = args@{...}: fasterrcnn.gen-feature-map-with-partial-image args;
          gen-feature-map = args@{...}: fasterrcnn.gen-feature-map args;
        };
        packages = {
          dataset = bdd100k;
          dataset-mini = bdd100k-mini;
          train = fasterrcnn.train {};
          finetuning = fasterrcnn.finetuning {};
          trainN = fasterrcnn.trainN;
          test = fasterrcnn.test {};
          detect = fasterrcnn.detect {};
          detect-sample = fasterrcnn.detect {
            datasets = img2coco  {
              dataset = sample-images;
            };
          };
          classification = fasterrcnn.classification {
            datasets = img2coco  {
              dataset = sample-image;
            };
          };
          gen-feature-map-with-partial-image = fasterrcnn.gen-feature-map-with-partial-image {};
          gen-feature-map = fasterrcnn.gen-feature-map {};
          inherit bdd100k;
        };

        # defaultPackage = self.packages.${system}.${packageName};

        devShell = fasterrcnn.myShell self system;
        
      });
}
