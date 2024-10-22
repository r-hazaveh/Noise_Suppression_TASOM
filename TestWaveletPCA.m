 
clc , clear all, close all

origImg =  (imread('lena.jpg'));
noisyImg = double(imnoise(origImg,'salt & pepper', 0.15));

X = noisyImg;   

%% Select a ROI, the size of which is not a power of 2. 
X = X(2:254,2:254);  

%% Display the original image. 
figure('color','w')
map = pink(255);
image(X)
colormap(map)
axis tight
title('Original Image')

%% Multilevel 2-D Non-Decimated Wavelet Decomposition

n = 3;                   % Decomposition Level 
w = 'sym4';               % Haar wavelet
thr = [70 70 70;70 70 70;70 70 40]% Coefficients for h d v* N;3*N

if size(thr,2)~= n
    error('Coefficients for h d v* N;3*N')
end
    
WT = ndwt2(X,n,w);        % Multilevel 2-D wavelet decomposition.
MaxLevel = wmaxlev(size(X),w)

%% Multilevel 2-D Non-Decimated Wavelet Coefficients
% Extract the resulting family of coefficients from the wavelet
% decomposition
cAA = cell(1,n);
cAD = cell(1,n);
cDA = cell(1,n);
cDD = cell(1,n);
for k = 1:n
    cAA{k} = indwt2(WT,'caa',k);   % Coefficients of approximations
    cAD{k} = indwt2(WT,'cad',k);   % Coefficients of horizontal details
    cDA{k} = indwt2(WT,'cda',k);   % Coefficients of vertical details
    cDD{k} = indwt2(WT,'cdd',k);   % Coefficients of diagonal details
end

%% Display the resulting family of coefficients 

figure('DefaultAxesXtick',[],'DefaultAxesYtick',[],'color','w')
colormap(map)
for k = 1:n
    subplot(4,n,k);
    imagesc(cAA{k}); xlabel(['cAA' int2str(k)])
    subplot(4,n,k+n);
    imagesc(abs(cAD{k})); xlabel(['cAD' int2str(k)])
    subplot(4,n,k+2*n);
    imagesc(abs(cDA{k})); xlabel(['cDA' int2str(k)])
    subplot(4,n,k+3*n);
    imagesc(abs(cDD{k})); xlabel(['cDD' int2str(k)])
end

%% Multilevel 2-D Non-Decimated Wavelet Reconstruction

A = cell(1,n);
D = cell(1,n);
for k = 1:n
    A{k} = indwt2(WT,'a',k);   % Approximations (low-pass components)
    D{k} = indwt2(WT,'d',k);   % Details (high-pass components)
end

% We now check that without changing the coefficients, the various
% reconstructions are perfect.
err = zeros(1,n);
for k = 1:n
    E = X-A{k}-D{k};
    err(k) = max(abs(E(:)));
end

%disp(error);
disp(['Error is : ' num2str(err) ]);
figure;
bar(err);

%%
figure('DefaultAxesXtick',[],'DefaultAxesYtick',[],'color','w')
colormap(map)
for k = 1:n
    subplot(2,n,k);   
    imagesc(A{k}); xlabel(['A' int2str(k)])
    subplot(2,n,k+n); 
    imagesc(abs(D{k})); xlabel(['Det' int2str(k)])
end

%%
figure('DefaultAxesXtick',[],'DefaultAxesYtick',[],'color','w')
colormap(map)
subplot(1,2,1);
image(X)
title('Original Image')
subplot(1,2,2);

%% Thresholding Section

c = 0;
for i = 2:3:n*3-1
    
    c = c +1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TempCoefh = WT.dec{i,1};
    [COEFF,SCORE] = princomp(TempCoefh);
    TempCoefh = SCORE;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for j = 1:size(TempCoefh,1)
    for k = 1:size(TempCoefh,2)
        
        if TempCoefh(j,k)<=thr(1,c)
            TempCoefh(j,k)= 0;
        else
            TempCoefh(j,k) = TempCoefh(j,k)-thr(1,c);
        end
        
    end
    end
    
   WT.dec{i,1} = TempCoefh;

end


c = 0;
 for i = 3:3:n*3
     
    c = c +1;
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TempCoefv = WT.dec{i,1};
    [COEFF,SCORE] = princomp(TempCoefv);
    TempCoefv = SCORE;
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    for j = 1:size(TempCoefv,1)
    for k = 1:size(TempCoefv,2)
        
        if TempCoefv(j,k)<=thr(2,c)
            TempCoefv(j,k)= 0;
        else
            TempCoefv(j,k) = TempCoefv(j,k)-thr(2,c);
        end
        
    end
    end
    
   WT.dec{i,1} = TempCoefv;

 end

c = 0;
for i = 4:3:n*3+1
    
    c = c + 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    TempCoefd = WT.dec{i,1};
    [COEFF,SCORE] = princomp(TempCoefd);
    TempCoefd = SCORE;    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for j = 1:size(TempCoefd,1)
    for k = 1:size(TempCoefd,2)
        
        if TempCoefd(j,k)<=thr(3,c)
            TempCoefd(j,k)= 0;
        else
            TempCoefd(j,k) = TempCoefd(j,k)-thr(3,c);
        end
        
    end
    end
    
   WT.dec{i,1} = TempCoefd;

end


%%
recX = indwt2(WT);
image(recX)
title('Denoised-PCA Image ')

%%  Calculate the parameter of PSNR 

dif_fil_org = abs(recX - X);
squre_dif = dif_fil_org.^2;
res = sum(squre_dif(:));

PSNR = 10 * log10((255^2)/((1/(size(X,1)*size(X,2)))* res));
fprintf('\n')
display(['The PSNR Parameter is : ', num2str(PSNR)]);

%%
% Consider the approximations at level 1 obtained using three different
% extension modes. These approximations are rough denoised images.
WTN = ndwt2(X,1,w,'mode','per');   % Multilevel 2-D wavelet decomposition.
A{1} = indwt2(WTN,'a',1);
WTN = ndwt2(X,1,w,'mode','sym');   % Multilevel 2-D wavelet decomposition.
A{2} = indwt2(WTN,'a',1);
WTN = ndwt2(X,1,w,'mode','spd');   % Multilevel 2-D wavelet decomposition.
A{3} = indwt2(WTN,'a',1);

%%
% As expected, the approximations are very similar, with the differences 
% concentrated on the edge.
figure('DefaultAxesXtick',[],'DefaultAxesYtick',[],'color','w')
colormap(map)
subplot(2,3,2);
image(X)
title('Original Image')
subplot(2,3,4);
image(A{1})
xlabel('A1 - PER')
subplot(2,3,5);
image(A{2})
title('Denoised Images')
xlabel('A1 - SYM')
subplot(2,3,6);
image(A{3})
xlabel('A1 - SPD')


