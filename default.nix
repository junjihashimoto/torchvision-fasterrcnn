{ pkgs
, poetry2nix
}:
with pkgs;
let
  customOverrides = self: super: {
    # Overrides go here
    opencv-python = super.opencv-python.overridePythonAttrs (old: {
      nativeBuildInputs = [ pkgs.cmake ] ++ old.nativeBuildInputs;
      buildInputs = [ self.scikit-build ] ++ (old.buildInputs or []);
      dontUseCmakeConfigure = true;
    });
  };

  mkDerivation = { pname, description, script, scriptArgs } : poetry2nix.mkPoetryApplication {
    inherit pname;
    version = "2021-06-27";
    projectDir = ./.;
      
    overrides = [ pkgs.poetry2nix.defaultPoetryOverrides customOverrides ];
    nativeBuildInputs = [
      curl
    ];
    buildInputs =  [];
    src = ./.;
    buildPhase = ''
      export CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
      #export REQUESTS_CA_BUNDLE=""
      export TRANSFORMERS_CACHE=$TMPDIR
      export XDG_CACHE_HOME=$TMPDIR
      mkdir output
      python ${script} \
        --output-dir "${scriptArgs.output}" 
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
in {
  train = mkDerivation {
    pname = "torchvision-fasterrcnn-trained";
    description = "Trained fasterrcnn";
    script = "train.py";
    scriptArgs = {
      output = "output";
    };
  };
}
