{ pkgs
, bdd100k
}:
let
  lib = pkgs.lib;
  myPython = pkgs.python3.withPackages (ps: with ps;
    [ opencv4
      pillow
      pytorch-bin
      torchvision-bin
      pycocotools
      numpy
    ]
  );
  mkDerivation = { pname
                 , description
                 , script
                 , scriptArgs
                 , pretrained ? ""
                 , numGpu
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
      bdd100k
    ];
    buildInputs =  [];
    src = ./src;
    buildPhase = ''
      export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
      #export REQUESTS_CA_BUNDLE=""
      export TRANSFORMERS_CACHE=$TMPDIR
      export XDG_CACHE_HOME=$TMPDIR

      export PIP_PREFIX=$(pwd)/_build/pip_packages
      export PYTHONPATH="$PIP_PREFIX/${myPython.sitePackages}:$PYTHONPATH"
      export PATH="$PIP_PREFIX/bin:$PATH"
      unset SOURCE_DATE_EPOCH
      mkdir output
      ln -s ${bdd100k.out} bdd100k
      if [ ${script} = "train.py" ] ; then 
        python -m torch.distributed.launch --nproc_per_node=${numGpu} --use_env \
          "${pretrained_str} \
          ${script} --output-dir "${scriptArgs.output}" --world-size ${numGpu} \
          --epochs "${scriptArgs.epochs}" \
          --lr "${scriptArgs.lr}"
      else
        python ${script} \
          --output-dir "${scriptArgs.output}" 
      fi
    '';
    installPhase = ''
      mkdir -p $out
      cp -r ${scriptArgs.output} $out
    '';
    meta = with lib; {
      inherit description;
      longDescription = ''
      '';
      homepage = "";
      license = licenses.bsd3;
      platforms = platforms.all;
      maintainers = with maintainers; [ junjihashimoto tscholak ];
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
      maintainers = with maintainers; [ junjihashimoto tscholak ];
    };
  };
  
in
rec {
  train = mkDerivation {
    pname = "torchvision-fasterrcnn-trained";
    description = "Trained fasterrcnn";
    # pretrained = checkpoint;
    numGpu = 3;
    script = "train.py";
    scriptArgs = {
      output = "output";
      lr = "0.12";
      epochs = "25";
    };
  };
}
