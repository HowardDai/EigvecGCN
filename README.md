# EigvecGCN
Repository with initial experiments using GCNs to predict Laplacian eigenvectors. 

## Environment setup 

```conda env create -f environment.yml```
```conda activate eigvecGCN```

## Run
Create a config.yml file with desired settings (see default_config.yml). Then run the following:
```python src/main.py --config src/config.yaml```

## Extra: Julia installation, for harmonic extension method

``` 
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.1-linux-x86_64.tar.gz
```

```
mkdir -p ~/julia
tar -xzf julia-1.10.1-linux-x86_64.tar.gz -C ~/julia
```

Set julia path to this julia (add this to ~/.bashrc to make this automatically happen every session)

```
export PATH="$HOME/julia/julia-1.10.1/bin:$PATH"
```

From terminal, install required packages:
``` 
julia
```

``` import Pkg
Pkg.add("PyCall")          
Pkg.build("PyCall")
Pkg.add("Laplacians")
```



