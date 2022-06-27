# HSICrop
1. Create Conda Environment
> $conda env create --file environment.yaml python=3.6

> $conda activate HSICrop

2. Prepare dataset
> install unzip on your machine (apt-get install unzip)
> 
> $sh download_data.sh
>
> Environmental information in env_info.txt as below order:
> 
>> "Filename" "Date" "Time" "Temperature" "Humidity" "Cloud level" "Flight altitude" "Solar altitude"
>> 
>> "Date" and "Time" indicate the difference from September 22, 2020 and 9:00, respectively, in days and minutes.
>>
>> For example, the first line in env_info.txt "Zone-A-001_013_038 0 59 22 60 4 150 40.91" means that Zone-A-001_013_038 image was acquired on September 22, 2020 at 9:59 AM. And the temperature at the time of acquisition was 22 degrees, the humidity was 60 percent, the cloud level was 4, the flight altitude was 150 meters, and the altitude of the sun was 40.91 degrees.

3. Edit Configure File (config.py)
> Set your machine's GPU ids to "GPUS".

4. Train Command
>  $python train.py --model HSIUNet

5. Test Command
>  $python test.py --model HSIUNet --weight "your_weight_path"

>  [link](https://drive.google.com/file/d/15GO76EHtr9RhUFCkf62nsvIqz02917Gg/view?usp=sharing): shared link to our trained weight file
