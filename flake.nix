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
#    nixpkgs.url = "github:NixOS/nixpkgs";
    nixpkgs.url = "github:junjihashimoto/nixpkgs?rev=71cda4fbef0c064b4df82ac65dd2cc868bb37c32";
    flake-utils.url = "github:numtide/flake-utils";
    hasktorch-datasets.url = "github:hasktorch/hasktorch-datasets";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
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
        fasterrcnn = pkgs.callPackage ./default.nix {bdd100k = hasktorch-datasets.packages.${system}.datasets-bdd100k-coco;};
      in {
        packages = {
          train = fasterrcnn.train;
          train2 = fasterrcnn.train2;
        };

        # defaultPackage = self.packages.${system}.${packageName};

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ poetry ];
          inputsFrom = builtins.attrValues self.packages.${system};
        };
      });
}
