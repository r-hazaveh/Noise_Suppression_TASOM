
clc , clear all, close all

% X contains the loaded image.
X = double(imread('noisyImg.jpg')); 

% Generate noisy image. 
x = X ;

% find default values (see ddencmp). 
[thr,sorh,keepapp] = ddencmp('den','wv',x);

%% Parameter of Denoising
w= 'sym4';
MaxLevel = wmaxlev(size(X),w);
keepapp = 0;
thr = [70 80;90 60;50 70];% Coefficients for h d v* N=2;
sorh = 's';

%display(['The Threshold is : ', num2str(thr)]);
display(['Type of Thresholding is : ', num2str(sorh)]);

% de-noise image using global thresholding option. 
xd = wdencmp('lvd',x,w,2,thr,sorh);% 'gbl' for global thresholding and we must use keepapp

%% Using some plotting commands,
% the following figure is generated.

figure(1);
imshow(uint8(x));
title('Noisy Image');

figure(2);
imshow(uint8(xd));
title('Denoised Image');

%%  Calculate the parameter of PSNR 

dif_fil_org = abs(xd - x);
squre_dif = dif_fil_org.^2;
res = sum(squre_dif(:));

PSNR = 10 * log10((255^2)/((1/(size(X,1)*size(X,2)))* res));
display(['The PSNR Parameter is : ', num2str(PSNR)]);

