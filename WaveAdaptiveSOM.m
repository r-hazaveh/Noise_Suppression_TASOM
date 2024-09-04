

clc , clear all, close all,warning off all

noisyImg = double(imread('noisyImg.jpg'));%X
origImg = imread('lena.jpg');
NoisyImgfirst = double(imnoise(origImg,'salt & pepper', 0.15));%Z
origImg = double(imread('lena.jpg'));%y

tmpOrigImg = origImg;

X =  noisyImg;   
Y =  origImg; 
Z =  NoisyImgfirst; 

%% Select a ROI, the size of which is not a power of 2. 
X = X(2:254,2:254);  
Y = Y(2:254,2:254);  
Z = Z(2:254,2:254);  

map = pink(255);

%% Multilevel 2-D Non-Decimated Wavelet Decomposition
% Perform non-decimated wavelet decomposition of signal X at level 4 using
% the haar wavelet.
n = 1;                   % Decomposition Level 
w = 'sym4';               % Haar wavelet
thr = 70 % Coefficients for h d v* N;3*N
    
WT = ndwt2(X,n,w);        % Multilevel 2-D wavelet decomposition.
WTO = ndwt2(Y,n,w); 
WTNF = ndwt2(Z,n,w); 

MaxLevel = wmaxlev(size(X),w)

%%
figure('DefaultAxesXtick',[],'DefaultAxesYtick',[],'color','w')
colormap(map)
subplot(1,2,1);
image(Z)
title('Noisy Image')
subplot(1,2,2);

%% Adaptive Thresholding Section

for I = 2:3:n*3-1
     
TempCoefh = WTO.dec{I,1};
origImg = TempCoefh;
origImg = double(origImg);

% Calculate 2 factors
PWMAD = zeros(size(origImg));
ROAD  = zeros(size(origImg));

for row = 3:size(origImg,1)-2
    for col = 3:size(origImg,2)-2
        
        tmpWin = origImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                
                Temp = origImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD(row,col) = sum(sortedDiffs(1:4));
        
    end
end

% remove one row and column and make IO
oneRowRemovedInput1 = PWMAD(2:end-1,2:end-1);
oneRowRemovedInput2 = ROAD(2:end-1,2:end-1);
inputs = [oneRowRemovedInput1(:)';oneRowRemovedInput2(:)']; %Important

%% make and train network
net1 = newsom(inputs,[5 8]);
net1.trainParam.epochs = 50;
net1 = train(net1,inputs);
Weight1 = net1.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions 

TempCoefh = WT.dec{I,1};
tstNoisyImg = TempCoefh;

% Calculate ROAD and PWMAD
PWMAD_tst = zeros(size(tstNoisyImg));
ROAD_tst  = zeros(size(tstNoisyImg));

for row = 3:size(tstNoisyImg,1)-2
    for col = 3:size(tstNoisyImg,2)-2
        
        tmpWin = tstNoisyImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst(row,col) = sum(sortedDiffs(1:4));
        
    end
 end

% remove one row and column and make IO
oneRowRemovedInput1_tst = PWMAD_tst(2:end-1,2:end-1);
oneRowRemovedInput2_tst = ROAD_tst(2:end-1,2:end-1);
inputs_tst = [oneRowRemovedInput1_tst(:)';oneRowRemovedInput2_tst(:)'];

%% make and train network
net2 = newsom(inputs_tst,[5 8]);
net2.trainParam.epochs = 50;
net2 = train(net2,inputs_tst);
Weight2 = net2.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions
TempCoefh = WTNF.dec{I,1};
tstNoisyImg2 = TempCoefh;

% Calculate ROAD and PWMAD
PWMAD_tst2 = zeros(size(tstNoisyImg2));
ROAD_tst2  = zeros(size(tstNoisyImg2));

for row = 3:size(tstNoisyImg2,1)-2
    for col = 3:size(tstNoisyImg2,2)-2
        
        tmpWin = tstNoisyImg2(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg2(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst2(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst2(row,col) = sum(sortedDiffs(1:4));
        
    end
end
% remove one row and column and make IO
oneRowRemovedInput1_tst2 = PWMAD_tst2(2:end-1,2:end-1);
oneRowRemovedInput2_tst2 = ROAD_tst2(2:end-1,2:end-1);
inputs_tst2 = [oneRowRemovedInput1_tst2(:) oneRowRemovedInput2_tst2(:)]; 

%% Classification

temp1 = zeros(size(Weight1,1),2);
temp2 = ones(size(Weight2,1),2);
Class = zeros(size(inputs_tst2,1),1);

for row = 1:size(inputs_tst2,1)
            
        for row2 = 1:size(Weight1,1)
            
            dist1 = (inputs_tst2(row,:)-Weight1(row2,:)).^2;
            temp1(row2,1) = (sum(dist1')).^0.5;  
            
        end
        
        for row2 = 1:size(Weight2,1)

            dist2 = (inputs_tst2(row,:)-Weight2(row2,:)).^2;
            temp2(row2,1) = (sum(dist2')).^0.5;  
            
        end       
            
        a = sort(temp1);
        b = sort(temp2);
        
        if a(1)< b(1)
            
            Class(row) = 0;
        else
            
            Class(row) = 1;
        end    
        
end

outputs = Class; 

%%
y_hat = outputs';
y_hat_matrix = reshape(y_hat,size(oneRowRemovedInput1_tst));

% add one row adn col to y_hat_matrix
tmpYhat = zeros(size(PWMAD_tst));
tmpYhat(2:end-1,2:end-1) = y_hat_matrix;

%% apply a simple median filter

tmpdec = WTNF.dec{I,1};
for row = 2:size(tmpdec,1)-1
    for col = 2:size(tmpdec,2)-1
        
        if tmpYhat(row,col) == 1      
         win = tmpdec(row-1:row+1,col-1:col+1);
            tmpdec(row,col) = median(win(:));
        end 
    end
end
   
WTNF.dec{I,1} = tmpdec;

end

%%%%%%%% Vertical

for I = 3:3:n*3
     
TempCoefv = WTO.dec{I,1};
   
origImg = TempCoefv;
origImg = double(origImg);

% Calculate 2 factors
PWMAD = zeros(size(origImg));
ROAD  = zeros(size(origImg));

for row = 3:size(origImg,1)-2
    for col = 3:size(origImg,2)-2
        
        tmpWin = origImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                
                Temp = origImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD(row,col) = sum(sortedDiffs(1:4));
        
    end
end

% remove one row and column and make IO
oneRowRemovedInput1 = PWMAD(2:end-1,2:end-1);
oneRowRemovedInput2 = ROAD(2:end-1,2:end-1);
inputs = [oneRowRemovedInput1(:)';oneRowRemovedInput2(:)']; %Important

%% make and train network
net1 = newsom(inputs,[5 8]);
net1.trainParam.epochs = 50;
net1 = train(net1,inputs);
Weight1 = net1.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions 

TempCoefv = WT.dec{I,1};
tstNoisyImg = TempCoefv;

% Calculate ROAD and PWMAD
PWMAD_tst = zeros(size(tstNoisyImg));
ROAD_tst  = zeros(size(tstNoisyImg));

for row = 3:size(tstNoisyImg,1)-2
    for col = 3:size(tstNoisyImg,2)-2
        
        tmpWin = tstNoisyImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst(row,col) = sum(sortedDiffs(1:4));
        
    end
 end

% remove one row and column and make IO
oneRowRemovedInput1_tst = PWMAD_tst(2:end-1,2:end-1);
oneRowRemovedInput2_tst = ROAD_tst(2:end-1,2:end-1);
inputs_tst = [oneRowRemovedInput1_tst(:)';oneRowRemovedInput2_tst(:)'];

%% make and train network
net2 = newsom(inputs_tst,[5 8]);
net2.trainParam.epochs = 50;
net2 = train(net2,inputs_tst);
Weight2 = net2.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions
TempCoefv = WTNF.dec{I,1};
tstNoisyImg2 = TempCoefv;

% Calculate ROAD and PWMAD
PWMAD_tst2 = zeros(size(tstNoisyImg2));
ROAD_tst2  = zeros(size(tstNoisyImg2));

for row = 3:size(tstNoisyImg2,1)-2
    for col = 3:size(tstNoisyImg2,2)-2
        
        tmpWin = tstNoisyImg2(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg2(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst2(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst2(row,col) = sum(sortedDiffs(1:4));
        
    end
end
% remove one row and column and make IO
oneRowRemovedInput1_tst2 = PWMAD_tst2(2:end-1,2:end-1);
oneRowRemovedInput2_tst2 = ROAD_tst2(2:end-1,2:end-1);
inputs_tst2 = [oneRowRemovedInput1_tst2(:) oneRowRemovedInput2_tst2(:)]; 

%% Classification

temp1 = zeros(size(Weight1,1),2);
temp2 = ones(size(Weight2,1),2);
Class = zeros(size(inputs_tst2,1),1);

for row = 1:size(inputs_tst2,1)
            
        for row2 = 1:size(Weight1,1)
            
            dist1 = (inputs_tst2(row,:)-Weight1(row2,:)).^2;
            temp1(row2,1) = (sum(dist1')).^0.5;  
            
        end
        
        for row2 = 1:size(Weight2,1)

            dist2 = (inputs_tst2(row,:)-Weight2(row2,:)).^2;
            temp2(row2,1) = (sum(dist2')).^0.5;  
            
        end       
            
        a = sort(temp1);
        b = sort(temp2);
        
        if a(1)< b(1)
            
            Class(row) = 0;
        else
            
            Class(row) = 1;
        end    
        
end

outputs = Class; 

%%
y_hat = outputs';
y_hat_matrix = reshape(y_hat,size(oneRowRemovedInput1_tst));

% add one row adn col to y_hat_matrix
tmpYhat = zeros(size(PWMAD_tst));
tmpYhat(2:end-1,2:end-1) = y_hat_matrix;

%% apply a simple median filter

tmpdec = WTNF.dec{I,1};
for row = 2:size(tmpdec,1)-1
    for col = 2:size(tmpdec,2)-1
        
        if tmpYhat(row,col) == 1      
            win = tmpdec(row-1:row+1,col-1:col+1);
            tmpdec(row,col) = median(win(:));
        end 
    end
end
   
WTNF.dec{I,1} = tmpdec;

end

%%%%%%%%%%%%%%%%%%%Diagonal

for I = 4:3:n*3+1
     
TempCoefd = WTO.dec{I,1};
origImg = TempCoefd;
origImg = double(origImg);

% Calculate 2 factors
PWMAD = zeros(size(origImg));
ROAD  = zeros(size(origImg));

for row = 3:size(origImg,1)-2
    for col = 3:size(origImg,2)-2
        
        tmpWin = origImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                
                Temp = origImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD(row,col) = sum(sortedDiffs(1:4));
        
    end
end

% remove one row and column and make IO
oneRowRemovedInput1 = PWMAD(2:end-1,2:end-1);
oneRowRemovedInput2 = ROAD(2:end-1,2:end-1);
inputs = [oneRowRemovedInput1(:)';oneRowRemovedInput2(:)']; %Important

%% make and train network
net1 = newsom(inputs,[5 8]);
net1.trainParam.epochs = 50;
net1 = train(net1,inputs);
Weight1 = net1.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions 

TempCoefd = WT.dec{I,1};
tstNoisyImg = TempCoefd;

% Calculate ROAD and PWMAD
PWMAD_tst = zeros(size(tstNoisyImg));
ROAD_tst  = zeros(size(tstNoisyImg));

for row = 3:size(tstNoisyImg,1)-2
    for col = 3:size(tstNoisyImg,2)-2
        
        tmpWin = tstNoisyImg(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst(row,col) = sum(sortedDiffs(1:4));
        
    end
 end

% remove one row and column and make IO
oneRowRemovedInput1_tst = PWMAD_tst(2:end-1,2:end-1);
oneRowRemovedInput2_tst = ROAD_tst(2:end-1,2:end-1);
inputs_tst = [oneRowRemovedInput1_tst(:)';oneRowRemovedInput2_tst(:)'];

%% make and train network
net2 = newsom(inputs_tst,[5 8]);
net2.trainParam.epochs = 50;
net2 = train(net2,inputs_tst);
Weight2 = net2.iw{1,1};

%% give a noisy image to ANN and get noisy pixel positions
TempCoefd = WTNF.dec{I,1};
tstNoisyImg2 = TempCoefd;

% Calculate ROAD and PWMAD
PWMAD_tst2 = zeros(size(tstNoisyImg2));
ROAD_tst2  = zeros(size(tstNoisyImg2));

for row = 3:size(tstNoisyImg2,1)-2
    for col = 3:size(tstNoisyImg2,2)-2
        
        tmpWin = tstNoisyImg2(row-1:row+1,col-1:col+1);
        
        for row2 = row-1:row+1
            for col2 = col-1:col+1
                Temp = tstNoisyImg2(row2-1:row2+1,col2-1:col2+1);
                
                for i = 1:size(tmpWin,1)
                    for j = 1:size(tmpWin,2)
                        M(i,j) = median(Temp(:));
                    end
                end
               
            end
        end
        
            
        % Calculate PWMAD
        m = median(tmpWin(:));
        d = abs(tmpWin(5) - m); 
        D = abs(tmpWin - M);
        PWMAD_tst2(row,col) = d - median(D(:));
        
        % Calculate ROAD4
        diffs = abs( tmpWin - tmpWin(5) );
        sortedDiffs = sort(diffs([1:4 6:9]));
        ROAD_tst2(row,col) = sum(sortedDiffs(1:4));
        
    end
end
% remove one row and column and make IO
oneRowRemovedInput1_tst2 = PWMAD_tst2(2:end-1,2:end-1);
oneRowRemovedInput2_tst2 = ROAD_tst2(2:end-1,2:end-1);
inputs_tst2 = [oneRowRemovedInput1_tst2(:) oneRowRemovedInput2_tst2(:)]; 

%% Classification

temp1 = zeros(size(Weight1,1),2);
temp2 = ones(size(Weight2,1),2);
Class = zeros(size(inputs_tst2,1),1);

for row = 1:size(inputs_tst2,1)
            
        for row2 = 1:size(Weight1,1)
            
            dist1 = (inputs_tst2(row,:)-Weight1(row2,:)).^2;
            temp1(row2,1) = (sum(dist1')).^0.5;  
            
        end
        
        for row2 = 1:size(Weight2,1)

            dist2 = (inputs_tst2(row,:)-Weight2(row2,:)).^2;
            temp2(row2,1) = (sum(dist2')).^0.5;  
            
        end       
            
        a = sort(temp1);
        b = sort(temp2);
        
        if a(1)< b(1)
            
            Class(row) = 0;
        else
            
            Class(row) = 1;
        end    
        
end

outputs = Class; 


%%

y_hat = outputs';
y_hat_matrix = reshape(y_hat,size(oneRowRemovedInput1_tst));

% add one row adn col to y_hat_matrix
tmpYhat = zeros(size(PWMAD_tst));
tmpYhat(2:end-1,2:end-1) = y_hat_matrix;

%% apply a simple median filter

tmpdec = WTNF.dec{I,1};
for row = 2:size(tmpdec,1)-1
    for col = 2:size(tmpdec,2)-1
        
        if tmpYhat(row,col) == 1      
         win = tmpdec(row-1:row+1,col-1:col+1);
            tmpdec(row,col) = median(win(:));
        end 
    end
end
   
WTNF.dec{I,1} = tmpdec;

end

% show Restored image

recX = indwt2(WTNF);

for i = 1:size(recX,1)
for j = 1:size(recX,2)
    recX(i,j) = 255*(recX(i,j)-min(recX(:)))/(max(recX(:))-min(recX(:)));
end
end

%image(recX)
imshow(uint8(recX))
title('Denoised-SOM2 Image ')

%%  Calculate the parameter of PSNR 

origImg = tmpOrigImg(2:254,2:254);
filteredImg = recX;

dif_fil_org = abs(filteredImg - origImg);
squre_dif = dif_fil_org.^2;
res = sum(squre_dif(:));

PSNR = 10 * log10((255^2)/((1/(size(filteredImg,1)*size(filteredImg,2)))* res));
display(['The PSNR Parameter is : ', num2str(PSNR)]);




