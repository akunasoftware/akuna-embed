{
  description = "akuna-embed dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        runtimeLibs =
          with pkgs;
          [ ]
          ++ lib.optionals stdenv.isLinux [
            libxkbcommon
            mesa
            vulkan-loader
            wayland
            libx11
            libxcursor
            libxi
            libxrandr
          ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            with pkgs;
            [
              curl
              gh
              git
              jq
              openssl
              pkg-config

              uv # python package manager for parity tests

              cargo # rust package manager
              cargo-deny # rust dependency license checker
              cargo-machete # rust dependency redundancy checker
              cargo-nextest # rust test runner
              clippy # rust linter
              rustc # rust compiler
              rust-analyzer # rust language server
              rustfmt # rust formatter
              sccache # rust compilation cache
            ]
            ++ runtimeLibs;

          env = {
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            RUSTC_WRAPPER = "${pkgs.sccache}/bin/sccache";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
            DYLD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.zstd ];
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          };
        };
      }
    );
}
