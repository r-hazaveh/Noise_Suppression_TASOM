
clc , clear all, close all

% X contains the loaded image.
X = double(imread('noisyImg.jpg')); 

% Generate noisy image. 
x = X ;

% Use wpdencmp for image de-noising. 
% Find default values (see ddencmp).
[thr,sorh,keepapp,crit] = ddencmp('den','wp',x);

%% Parameter of Denoising
w = 'sym4';
MaxLevel = wmaxlev(size(X),w);
crit = 'threshold';
keepapp = 0;
thr = 70;
sorh = 's';
display(['The Threshold is : ', num2str(thr)]);
display(['Type of Thresholding is : ', num2str(sorh)]);

% De-noise image using global thresholding with 
% SURE best basis. 
[xd,treed] = wpdencmp(x,sorh,3,w,crit,thr,keepapp);

%% Using some plotting commands,
% the following figure is generated

figure(1);
imshow(uint8(x));
title('Noisy Image');

figure(2);
imshow(uint8(xd));
title('Denoised Image');

plot(treed);

%%  Calculate the parameter of PSNR 

dif_fil_org = abs(xd - x);
squre_dif = dif_fil_org.^2;
res = sum(squre_dif(:));

PSNR = 10 * log10((255^2)/((1/(size(X,1)*size(X,2)))* res));
display(['The PSNR Parameter is : ', num2str(PSNR)]);

