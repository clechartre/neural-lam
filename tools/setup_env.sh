#!/bin/bash
#
# Create conda environment with pinned or unpinned requirements

if [[ ${BASH_SOURCE[0]} != "${0}"   ]]; then
    echo "Please simply call the script instead of sourcing it!"
    return
fi

# Default env names
DEFAULT_ENV_NAME="neural-lam"

# Default options
ENV_NAME="${DEFAULT_ENV_NAME}"
PINNED=true
EXPORT=false
CONDA=mamba
HELP=false

help_msg="Usage: $(basename "${0}") [-n NAME] [-p VER] [-u] [-e] [-m] [-h]

Options:
 -n NAME    Env name [default: ${DEFAULT_ENV_NAME}
 -u         Use unpinned requirements (minimal version restrictions)
 -e         Export environment files (requires -u)
 -m         Use mamba instead of conda
 -h         Print this help message and exit
"

# Eval command line options
while getopts n:p:defhimu flag; do
    case ${flag} in
        n) ENV_NAME=${OPTARG} ;;
        e) EXPORT=true ;;
        h) HELP=true ;;
        m) CONDA=mamba ;;
        u) PINNED=false ;;
        ?)
            echo -e "\n${help_msg}" >&2
                                        exit 1
                                              ;;
    esac
done

if ${HELP}; then
    echo "${help_msg}"
    exit 0
fi

echo "Setting up environment for installation"
eval "$(conda shell.bash hook)" || exit  # NOT ${CONDA} (doesn't work with mamba)
conda activate || exit # NOT ${CONDA} (doesn't work with mamba)

# Install requirements in new env
if ${PINNED}; then
    echo "Pinned installation"
    ${CONDA} env create --name ${ENV_NAME} --file requirements/environment.yml || exit
else
    echo "Unpinned installation"
    ${CONDA} env create --name ${ENV_NAME} --file requirements/requirements.yml || exit
    if ${EXPORT}; then
        echo "Export pinned prod environment"
        ${CONDA} env export --name ${ENV_NAME} --no-builds | \grep -v '^prefix:' >requirements/environment.yml  || exit
    fi
fi
