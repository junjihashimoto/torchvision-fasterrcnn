{ pkgs
, bdd100k
, bdd100k-mini
, hasktorch-datasets-utils
}:
let
  lib = pkgs.lib;
  resnet50 = builtins.fetchurl {
    url = "https://download.pytorch.org/models/resnet50-0676ba61.pth";
    sha256 = "12pv4bkq7rn9yy9njjrq9mi9gqp5hac8bzfgfcbvwnvrnrhvlxh6";
  };
  
  patched-pycocotools = pkgs.python39Packages.pycocotools.overrideAttrs (old: rec{
    patches = [./patches/pycocotools.patch];
  });
  myPython = pkgs.python39.withPackages (ps: with ps;
    [ opencv4
      pillow
      pytorch-bin
      torchvision-bin
      patched-pycocotools
      numpy
    ]
  );
  mkDerivation = { pname
                 , description
                 , script
                 , scriptArgs
                 , pretrained ? ""
                 , numGpu
                 , datasets
                 } :
                   let pretrained_str =
                         if pretrained == ""
                         then ""
                         else " --resume ${pretrained.out}/output/checkpoint.pth";
                   in  pkgs.stdenv.mkDerivation {
    pname = pname;
    version = "1";
    nativeBuildInputs = [
      myPython
      pkgs.curl
      datasets
    ];
    buildInputs =  [];
    src = hasktorch-datasets-utils.excludeFiles 
      [ "^test\.py$"
        "^inference\.py$"
      ]
      ./src;
    buildPhase = ''
      export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
      #export REQUESTS_CA_BUNDLE=""
      export TRANSFORMERS_CACHE=$TMPDIR
      export XDG_CACHE_HOME=$TMPDIR

      export PIP_PREFIX=$(pwd)/_build/pip_packages
      export PYTHONPATH="$PIP_PREFIX/${myPython.sitePackages}:$PYTHONPATH"
      export PATH="$PIP_PREFIX/bin:$PATH"
      unset SOURCE_DATE_EPOCH
      mkdir -p ../torch/hub/checkpoints
      ln -s ${resnet50} ../torch/hub/checkpoints/resnet50-0676ba61.pth
      ls -l ../torch/hub/checkpoints
      mkdir output
      ln -s ${datasets.out} bdd100k
      python -m torch.distributed.launch --nproc_per_node=${toString numGpu} --use_env \
        ${script} \
        ${pretrained_str} \
        ${expandStriptArgs scriptArgs}
    '';
    installPhase = ''
      mkdir -p $out
      cp -r ${scriptArgs.output-dir} $out
    '';
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto ];
    };
  };
  testDerivation = { pname
         , description
         , script
         , scriptArgs
         , pretrained
         , datasets
         } :
           let pretrained_str = " --resume ${pretrained.out}/output/model.pth";
           in  pkgs.stdenv.mkDerivation {
    pname = pname;
    version = "1";
    nativeBuildInputs = [
      myPython
      pkgs.curl
      pretrained
      datasets
    ];
    buildInputs =  [];
    src = hasktorch-datasets-utils.excludeFiles 
      [ "^train\.py$"
        "^inference\.py$"
      ]
      ./src;
    buildPhase = ''
      export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
      #export REQUESTS_CA_BUNDLE=""
      export TRANSFORMERS_CACHE=$TMPDIR
      export XDG_CACHE_HOME=$TMPDIR

      export PIP_PREFIX=$(pwd)/_build/pip_packages
      export PYTHONPATH="$PIP_PREFIX/${myPython.sitePackages}:$PYTHONPATH"
      export PATH="$PIP_PREFIX/bin:$PATH"
      unset SOURCE_DATE_EPOCH
      pwd
      mkdir -p ../torch/hub/checkpoints
      ln -s ${resnet50} ../torch/hub/checkpoints/resnet50-0676ba61.pth
      ls -l ../torch/hub/checkpoints
      mkdir output
      ln -s ${datasets.out} bdd100k
      python ${script} \
        ${pretrained_str} \
        --output-dir "${scriptArgs.output}" \
        2>&1 | tee test.log
      python log2json.py test.log
    '';
    installPhase = ''
      mkdir -p $out
      cp -r ${scriptArgs.output} $out
      cp test.log $out/
      cp map_results.json $out/
    '';
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto ];
    };
  };
  expandStriptArgs = scriptArgs:
    let names = builtins.attrNames scriptArgs;
        args = builtins.map (n:
          "--" + n + " " +
          "\"" +
          builtins.toString (scriptArgs."${n}") +
          "\""
        ) names;
    in builtins.concatStringsSep " " args;
  detectDerivation = { pname
         , description
         , script
         , scriptArgs
         , pretrained
         , datasets
         } :
           let pretrained_str = " --resume ${pretrained.out}/output/model.pth";
           in  pkgs.stdenv.mkDerivation {
    pname = pname;
    version = "1";
    nativeBuildInputs = [
      myPython
      pkgs.curl
      pretrained
      datasets
    ];
    buildInputs =  [];
    src = hasktorch-datasets-utils.excludeFiles 
      [ "^train\.py$"
        "^test\.py$"
      ]
      ./src;
    buildPhase = ''
      export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
      #export REQUESTS_CA_BUNDLE=""
      export TRANSFORMERS_CACHE=$TMPDIR
      export XDG_CACHE_HOME=$TMPDIR

      export PIP_PREFIX=$(pwd)/_build/pip_packages
      export PYTHONPATH="$PIP_PREFIX/${myPython.sitePackages}:$PYTHONPATH"
      export PATH="$PIP_PREFIX/bin:$PATH"
      unset SOURCE_DATE_EPOCH
      pwd
      mkdir -p ../torch/hub/checkpoints
      ln -s ${resnet50} ../torch/hub/checkpoints/resnet50-0676ba61.pth
      ls -l ../torch/hub/checkpoints
      mkdir output
      ln -s ${datasets.out} bdd100k
      python ${script} \
        ${pretrained_str} \
        ${expandStriptArgs scriptArgs}
    '';
    installPhase = ''
      mkdir -p $out
      cp -r ${scriptArgs.output-dir} $out
    '';
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto ];
    };
  };
  checkpoint = pkgs.stdenv.mkDerivation {
    pname = "checkpoint";
    version = "1";
    src = builtins.fetchurl {
        "sha256"= "0jiir49zhc3m9w2d5d3wyzpl6lrrv294jqy4jllq322875avjx38";
        "url"= "file:///home/hashimoto/git/torchvision-fastercnn-train/01/output/checkpoint.pth";
    };
    unpackCmd = ''
      mkdir -p $out/output
      cp "$curSrc" $out/output/"$'' + ''{curSrc#*-}"
      sourceRoot=`pwd`
    '';
    dontFixup = true;
    dontInstall = true;
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto ];
    };
  };
  pretrainedModel = pkgs.stdenv.mkDerivation {
    pname = "pretrained-fasterrcnn";
    version = "1";
    src = builtins.fetchurl {
        "sha256"= "1xhx9xiw5gnlnnz6kgjrarwvp5b1qvpni0sx8m99v5g1vjnibfn3";
        "url"= "https://github.com/hasktorch/hasktorch-datasets/releases/download/bdd100k/torchvision_fasterrcnn_model_e83ca1b.pth";
    };
    unpackCmd = ''
      mkdir -p $out/output
      cp "$curSrc" $out/output/model.pth
      sourceRoot=`pwd`
    '';
    dontFixup = true;
    dontInstall = true;
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto ];
    };
  };
  iota = n: start:
    if n == 0
    then []
    else [start] ++ iota (n - 1) (start+1);
  
in
rec {
  train = args@{...} : mkDerivation (rec {
    pname = "torchvision-fasterrcnn-trained";
    description = "Trained fasterrcnn";
    # pretrained = checkpoint;
    numGpu = 3;
    datasets = bdd100k;
    script = "train.py";
    scriptArgs = rec {
      output-dir = "output";
      epochs = 26;
      world-size = numGpu;
      batch-size = 12;
      # https://arxiv.org/abs/1711.00489
      lr = 0.02 * (batch-size / 2.0);
      # momentum = 0.9 * (1.0-(1.0/(batch-size / 2.0)));
    };
  } // args);
  trainN =
    let numGpu = 3;
        scriptArgs' = epoch: rec {
          output-dir = "output";
          epochs = epoch;
          world-size = numGpu;
          batch-size = 12;
          # https://arxiv.org/abs/1711.00489
          lr = 0.02 * (batch-size / 2.0);
          # momentum = 0.9 * (1.0-(1.0/(batch-size / 2.0)));
        };
    in builtins.foldl'
      (prev: epoch:
        train {
          pretrained = prev;
          scriptArgs = scriptArgs' epoch;
        }
      ) (train {
        scriptArgs = scriptArgs' 1;
      }
      ) (iota 26 2);
  test = args@{...} : testDerivation ({
    pname = "torchvision-fasterrcnn-test";
    description = "The test of fasterrcnn";
    script = "test.py";
    scriptArgs = {
      output = "output";
    };
    pretrained = pretrainedModel;
    datasets = bdd100k;
  } // args);
  detect = args@{...} : detectDerivation ({
    pname = "torchvision-fasterrcnn-detect";
    description = "The inference of fasterrcnn";
    script = "inference.py";
    scriptArgs = {
      device = "cuda";
      output-dir = "output";
    };
    pretrained = pretrainedModel;
    datasets = bdd100k-mini;
  } // args);
}
