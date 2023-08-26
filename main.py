#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:42:51 2023

@author: ericwei
"""
import sys
sys.path.append('Experiments/Experiment1_BenchMark')
import Experiment1_BenchMark as ex1
sys.path.append('Experiments/Experiment2_Compression')
import Experiment2_Compression as ex2
sys.path.append('Experiments/Experiment3_Noises')
import Experiment3_GaussianNoise as ex3g
import Experiment3_MotionBlur as ex3m
import Experiment3_SaltAndPepper as ex3s
import Experiment3_ColourVariation as ex3c
import Experiment3_BrightnessVariation as ex3b
sys.path.append('Experiments/Experiment4_NoiseCompression')
import Experiment4_Gaussian as ex4g
import Experiment4_MotionBlur as ex4m
import Experiment4_SaltAndPepper as ex4s
import Experiment4_ColourVariation as ex4c
import Experiment4_BrightnessVariation as ex4b
sys.path.append('Experiments/Experiment5_NoiseCombination')
import Experiment5_ColourandGaussian as ex5cg
import Experiment5_ColourandMotion as ex5cm
import Experiment5_ColourandSalt as ex5cs
import Experiment5_BrightnessAndGaussian as ex5bg
import Experiment5_BrightnessAndMotion as ex5bm
import Experiment5_BrightnessAndSalt as ex5bs
sys.path.append('Experiments/Experiment6_Finetune')
import Experiment6_Finetune as ex6
sys.path.append('Experiments/Experiment7_Novel_Compression_Model')
import ML_Compression_Model as ex7



data_dir = 'Dataset/101_ObjectCategories'
model_path = 'FineTuned_Model/MobileNet_final_model.h5'
# ex1.run_experiment_one(data_dir)
# ex2.run_experiment_two(data_dir)
# ex3g.run_experiment_three_gaussian(data_dir)
# ex3m.run_experiment_three_motion(data_dir)
# ex3s.run_experiment_three_salt(data_dir)
# ex3c.run_experiment_three_colour(data_dir)
# ex3b.run_experiment_three_brightness(data_dir)
# ex4g.run_experiment_four_gaussian(data_dir)
# ex4m.run_experiment_four_motion(data_dir)
# ex4s.run_experiment_four_salt(data_dir)
# ex4c.run_experiment_four_colour(data_dir)
# ex4b.run_experiment_four_brightness(data_dir)
# ex5cg.run_experiment_five_colour_gaussian(data_dir)
# ex5cm.run_experiment_five_colour_motion(data_dir)
# ex5cs.run_experiment_five_colour_salt(data_dir)
# ex5bg.run_experiment_five_brightness_gaussian(data_dir)
# ex5bm.run_experiment_five_brightness_motion(data_dir)
# ex5bs.run_experiment_five_brightness_salt(data_dir)
# result = ex6.run_experiment_six(data_dir, model_path)
ex7.run_experiment_seven(data_dir, model_path)