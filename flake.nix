{
  description = "torchvision-fasterrcnn";

  nixConfig = {
    substituters = [
      # https://iohk.cachix.org
      https://hydra.iohk.io
    ];
    trusted-public-keys = [
      hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ=
    ];
    # bash-prompt = "toto";
  };
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?rev=8b0f315b7691adcee291b2ff139a1beed7c50d94";
    flake-utils.url = "github:numtide/flake-utils";
    hasktorch-datasets.url = "github:hasktorch/hasktorch-datasets";
    poetry2nix = {
      url = "github:nix-community/poetry2nix?rev=046bddd753f99ce73ab2fbb91a2d5f620eadae39";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, hasktorch-datasets, poetry2nix }:
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
        fasterrcnn = pkgs.callPackage ./default.nix {
          inherit bdd100k;
          inherit bdd100k-mini;
          hasktorch-datasets-utils = hasktorch-datasets.lib.${system}.utils;
        };
      in {
        lib = {
          train = args@{...}: fasterrcnn.train args;
          test = args@{...}: fasterrcnn.test args;
          detect = args@{...}: fasterrcnn.detect args;
        };
        packages = {
          train = fasterrcnn.train {};
          test = fasterrcnn.test {};
          detect = fasterrcnn.detect {};
          detect-sample = fasterrcnn.detect {
            datasets = img2coco  {
              dataset = sample-images;
            };
          };
          inherit bdd100k;
        };

        # defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ poetry ];
          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}
