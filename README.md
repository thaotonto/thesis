# Automatic Vietnamese bike's plate license recognition system

![preview](preview.png)
## installation
install `Anaconda` **[here](https://www.anaconda.com/distribution/)**

Create a new virtual environment called `darkflow` or any name you like
```
conda create -n darkflow pip python=3.6
```
Then, activate the environment by issuing:
```
activate darkflow
```
Install tensorflow-cpu in this environment by issuing:
```
pip install --ignore-installed --upgrade tensorflow
```
Install the other necessary packages by issuing the following commands:
```
conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install matplotlib
pip install pandas
pip install opencv-python
```
clone darkflow repo and built it by run these command 

```bash
git clone https://github.com/thtrieu/darkflow
cd darkflow
pip install -e .
```

Clone this repo
```bash
git clone https://github.com/thaotonto/thesis.git
```

download trained weight [here](https://drive.google.com/open?id=1iWgwIc23nIIFyC0WPS_asPZ6GCBVJqXM) and extract it to `darkflow/`

## Run the system
make sure you have a webcam on you laptop/computer

run this command to test the system 
```bash
python plate_system -vid demo.mov
```

**note**: make sure you run the right python if you run into error like missing tensorflow try again with this command

```bash
python3 plate_system -vid demo.mov
```
