# HSICrop

1. Create Conda Environment
> $conda env create --file environment.yaml python=3.6

2. Prepare dataset
> $sh download_data.sh

3. Edit Configure File (config.py)
> Set your machine's GPU ids to "GPUS".

4. Train Command
>  $python train.py --model HSIUNet

5. Test Command
>  $python test.py --model HSIUNet --weight "your_weight_path"

>  [link](https://drive.google.com/file/d/15GO76EHtr9RhUFCkf62nsvIqz02917Gg/view?usp=sharing): shared link to our trained weights file
