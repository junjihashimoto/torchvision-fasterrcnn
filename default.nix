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
  mkDerivation = { pname, description, script, scriptArgs } : pkgs.stdenv.mkDerivation {
    pname = pname;
    version = "1";
    nativeBuildInputs = [
      myPython
      pkgs.curl
      bdd100k
    ];
    buildInputs =  [];
    src = ./.;
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
        python -m torch.distributed.launch --nproc_per_node=3 --use_env \
          ${script} --output-dir "${scriptArgs.output}" --world-size 3
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
  mkDerivation2 = { pname, description, script, scriptArgs } : pkgs.stdenv.mkDerivation {
    pname = pname;
    version = "1";
    nativeBuildInputs = [
      myPython
      pkgs.curl
      bdd100k
    ];
    buildInputs =  [];
    src = ./.;
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
        python -m torch.distributed.launch --nproc_per_node=3 --use_env \
          ${script} --output-dir "${scriptArgs.output}" --world-size 3
      else
        python ${script} \
          --resume "${scriptArgs.resume}" 
          --epochs "${scriptArgs.epochs}" 
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
in
{
  train = mkDerivation {
    pname = "torchvision-fasterrcnn-trained";
    description = "Trained fasterrcnn";
    script = "train.py";
    scriptArgs = {
      output = "output";
    };
  };
  train2 = mkDerivation2 {
    pname = "torchvision-fasterrcnn-trained";
    description = "Trained fasterrcnn";
    script = "train.py";
    scriptArgs = {
      epochs = "56";
      resume = "checkpoint.pth";
      output = "output";
    };
  };
}
