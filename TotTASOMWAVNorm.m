
clc, clear all, close all , warning off all

global numberOfNeuron

n=1;
w='sym4';
origImg = imread('lena.jpg');
origImg = double(origImg);

% clip maximum values if any
origImg(find(origImg>255)) = 255;
% clip minimum values if any
origImg(find(origImg<0)) = 0;

% Calculate 2 factors
PWMAD = zeros(size(origImg));
ROAD  = zeros(size(origImg));

for row = 5:size(origImg,1)-4
    for col = 5:size(origImg,2)-4
        
        
        TempWav1 = origImg(row-4:row+4,col-4:col+4);
        TempWav2 = ndwt2(TempWav1,n,w);
        tmpWin = TempWav2.dec{1,1};
        
        % Calculate Std
        TempVar2 = median(tmpWin);  
        PWMAD(row,col) = median(TempVar2');
        
        % Calculate Mean
        
        ROAD(row,col) = norm(tmpWin,'fro');
        
    end
end

    %normalize 
    TempN = PWMAD;
    
    for r=1:size(PWMAD,2)         
        if(std(TempN(:,r))== 0)           
                PWMAD(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                PWMAD(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end  

      TempN = ROAD;
    
    for r=1:size(ROAD,2)         
        if(std(TempN(:,r))== 0)           
                ROAD(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                ROAD(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end    
    
    
    
% remove one row and column and make IO
oneRowRemovedInput1 = PWMAD(2:end-1,2:end-1);
oneRowRemovedInput2 = ROAD(2:end-1,2:end-1);
inputs = [oneRowRemovedInput1(:) oneRowRemovedInput2(:)]; %Important

%% make and train network

    global Weight
    niteration = 4;
    L1 = TasomNeuralNetwok2(inputs(766:63751,:),niteration);
    WeightCom1 = Weight;
    
  

%% give a noisy image to ANN and get noisy pixel positions
tstImg = ( imread('lena.jpg') );
tstNoisyImg = double(tstImg);
tstNoisyImg(3:254,3:254) = double(imnoise(tstImg(3:254,3:254),'salt & pepper', 0.15));

% Calculate ROAD and PWMAD
PWMAD_tst = zeros(size(tstNoisyImg));
ROAD_tst  = zeros(size(tstNoisyImg));

for row = 5:size(tstNoisyImg,1)-4
    for col = 5:size(tstNoisyImg,2)-4
        
        TempWav1 = tstNoisyImg(row-4:row+4,col-4:col+4);
        TempWav2 = ndwt2(TempWav1,n,w);
        tmpWin = TempWav2.dec{1,1};
        
        % Calculate Std
        TempVar2 = median(tmpWin);  
        PWMAD_tst(row,col) = median(TempVar2');
        
        % Calculate Mean
        ROAD_tst(row,col) = norm(tmpWin,'fro');
        
    end
end
 
        
    %normalize 
    TempN = PWMAD_tst;
    
    for r=1:size(PWMAD_tst,2)         
        if(std(TempN(:,r))== 0)           
                PWMAD_tst(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                PWMAD_tst(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end
    
    
    TempN = ROAD_tst;
    
    for r=1:size(ROAD_tst,2)         
        if(std(TempN(:,r))== 0)           
                ROAD_tst(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                ROAD_tst(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end         
        
        
% remove one row and column and make IO
oneRowRemovedInput1_tst = PWMAD_tst(2:end-1,2:end-1);
oneRowRemovedInput2_tst = ROAD_tst(2:end-1,2:end-1);
inputs_tst = [oneRowRemovedInput1_tst(:) oneRowRemovedInput2_tst(:)];

%% make and train network

    L2 = TasomNeuralNetwok2(inputs_tst(766:63751,:),niteration);
    WeightCom2 = Weight;
    
%% give a noisy image to ANN and get noisy pixel positions
tstImg2 = ( imread('lena.jpg') );
tstNoisyImg2 = double(tstImg2);
tstNoisyImg2(3:254,3:254) = double(imnoise(tstImg2(3:254,3:254),'salt & pepper', 0.15));

% show noisy image
figure(1);
imshow(uint8(tstNoisyImg2))
title 'Noisy Image'

% Calculate ROAD and PWMAD
PWMAD_tst2 = zeros(size(tstNoisyImg2));
ROAD_tst2  = zeros(size(tstNoisyImg2));

for row = 5:size(tstNoisyImg2,1)-4
    for col = 5:size(tstNoisyImg2,2)-4
        
        TempWav1 = tstNoisyImg2(row-4:row+4,col-4:col+4);
        TempWav2 = ndwt2(TempWav1,n,w);
        tmpWin = TempWav2.dec{1,1};
        
       
        % Calculate Std
        TempVar2 = median(tmpWin);  
        PWMAD_tst2(row,col) = median(TempVar2');
        
        % Calculate Mean
        
        ROAD_tst2(row,col) = norm(tmpWin,'fro');
      
       
    end
end

    %normalize 
    TempN = PWMAD_tst2;
    
    for r=1:size(PWMAD_tst2,2)         
        if(std(TempN(:,r))== 0)           
                PWMAD_tst2(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                PWMAD_tst2(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end
    
    
    TempN = ROAD_tst2;
    
    for r=1:size(ROAD_tst2,2)         
        if(std(TempN(:,r))== 0)           
                ROAD_tst2(:,r)=TempN(:,r);
        end
        if(std(TempN(:,r))> 0) 
                ROAD_tst2(:,r)=(TempN(:,r)-mean(TempN(:,r)))/std(TempN(:,r));
        end   
    end 





% remove one row and column and make IO
oneRowRemovedInput1_tst2 = PWMAD_tst2(2:end-1,2:end-1);
oneRowRemovedInput2_tst2 = ROAD_tst2(2:end-1,2:end-1);
inputs_tst2 = [oneRowRemovedInput1_tst2(:) oneRowRemovedInput2_tst2(:)]; 

%% Classification

    Weight1 = WeightCom1;    
    Weight2 = WeightCom2;
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

filteredImg = tstNoisyImg2;
for row = 2:size(tstNoisyImg,1)-1
    for col = 2:size(tstNoisyImg,2)-1
        if tmpYhat(row,col) == 1           
            win = tstNoisyImg2(row-1:row+1,col-1:col+1);
            filteredImg(row,col) = median(win(:));
        end       
    end
end

% show Restored image
figure(2);
imshow(uint8(filteredImg))
title 'Restored Image'


%%  Calculate the parameter of PSNR 

dif_fil_org = abs(filteredImg - origImg);
squre_dif = dif_fil_org.^2;
res = sum(squre_dif(:));

PSNR = 10 * log10((255^2)/((1/(size(filteredImg,1)*size(filteredImg,2)))* res));
display(['The PSNR Parameter is : ', num2str(PSNR)]);      
  





