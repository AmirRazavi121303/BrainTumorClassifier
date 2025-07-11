I decided to test my limits and make a brain tumor classifier based on a dataset from kaggle titled "brain mri scans"
the premise is to train a model on pytorch to classify between four different types of brain MRI's:
1. Meningioma, 2. Glioma, 3. Pituitary, 4. No Tumor

This is a big project for me as I had never before used pytorch and have no knowledge within cancer biology. 
I started off by watching a classic pytorch tutorial just to get an idea of the syntax and how the model itself works
I then decided to create my own algorithm side by side with this tutorial
this proved to be very good since it taught me a lot of the basic architecture
however after noticing the model was not learning anything and that the predictions on the test data were coming out purely random, i had to take a step back

after lots and lots of tweaking the model, nothing changed (if anything it got worse)

I decided to remake the entire program without any tutorial, going purely off the things i had already written and what i know about ML
this was an amazing excersize as I was commenting what each line does and it helped me get a much stronger grasp for the programming aspect of it
after battling with this for a while, I got everything to work.

some things I learned while doing this:
- image processing
- pytorch
- more complex ML engineering
- kaggle
- tweaking parameters to achieve optimal losses

some things I need to work on for my next projects:
- visualization (not familiar enough with matplotlib)
- validation data
- different types of models
- how different libraries colaborate
- domain knowledge (in this case cancer biology)
- scaling and making it more practically useful

------regarding the training loop: --------

I worked for a while on trying to get it as close as possible to perfect. Heres some things I tried:
- Adam instead of SGD for the optimizer
- switching from resnet50 to 18
- raising epochs from 20 to 50
- adding a whole bunch of if loops to make sure its training correctly

All but one of these changes ended up working, the one being using Adam. I found that Adam was fluctuating way to much (it would go from a val_loss of 0.09 to 8.1)
Ultimately I prefer SGD since it was much more controlled and it was taking slower but gradual steps. I ran it with an epoch of 50 that saved it whenever the losses
got below a certain threshold.

Things I learned:
- tweaking parameters in order to find a minimum
- creating custom training loops

Things I should work on for the future:
- using adam more efficiently
- writing the code alot cleaner

to get an idea of the type of losses I was getting toward the end:

Epoch 40/50 - Train Loss: 0.0176, Val Loss: 0.0916
Epoch 41/50 - Train Loss: 0.0158, Val Loss: 0.0821
Epoch 42/50 - Train Loss: 0.0148, Val Loss: 0.0835
Epoch 43/50 - Train Loss: 0.0169, Val Loss: 0.0923
Epoch 44/50 - Train Loss: 0.0148, Val Loss: 0.0937
Epoch 45/50 - Train Loss: 0.0151, Val Loss: 0.0864
Epoch 46/50 - Train Loss: 0.0134, Val Loss: 0.0825
Epoch 47/50 - Train Loss: 0.0121, Val Loss: 0.0882
Epoch 48/50 - Train Loss: 0.0131, Val Loss: 0.0890
Epoch 49/50 - Train Loss: 0.0113, Val Loss: 0.0837
Epoch 50/50 - Train Loss: 0.0124, Val Loss: 0.0804 <----- this is the one i'm using in the applying_model.py

I understand I could've gotten better results with more epochs, but I feel like this is pretty good for my first try at this


This was a very fun project and I will be definitely "throwing" myself at more difficult projects in the future. 
I am a firm believer that doing something complex is the best way to learn something and this project proved it to me as I come out way stronger and more confident
