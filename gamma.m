Patch_path='data/bright_dark_pixel_patch/';
Patch_gamma_path='data/bright_dark_pixel_patch-gamma/';
dirOutput=dir(Patch_path);
dirOutput=dirOutput(3:length(dirOutput));
tic
for i=1:length(dirOutput)
    %length(dirOutput)
    I=imread(strcat(Patch_path,dirOutput(i).name));
    %I=imresize(I,1);
    L1=I(:,:,1);
    L2=I(:,:,2);
    L3=I(:,:,3);
    J1=imadjust(L1,[],[],1/2.2);
    J2=imadjust(L2,[],[],1/2.2);
    J3=imadjust(L3,[],[],1/2.2);
    K=I;
    K(:,:,1)=J1;
    K(:,:,2)=J2;
    K(:,:,3)=J3;
    imwrite(K,strcat(Patch_gamma_path,dirOutput(i).name));
    if rem(i,10)==0
        fprintf('%d\n',i)
        toc
    end    
end
