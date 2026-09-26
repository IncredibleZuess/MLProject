{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{

  # Packages needed for PyTorch, OpenCV, CUDA, and build tools on NixOS
  packages = [
    pkgs.git
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.glib
    pkgs.zlib
  ];

  enterShell = ''
    export LD_LIBRARY_PATH="/run/current-system/sw/share/nix-ld/lib:/run/opengl-driver/lib:${lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.libGL pkgs.glib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;



  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}" 
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
