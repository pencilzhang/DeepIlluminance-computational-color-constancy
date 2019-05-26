

dirOutput=dir(['data/colorchecker-origin/' '*.png']);
for i=1:568
    image=imread(['data/colorchecker-origin/' dirOutput(i).name]);
    I=uint8(image/16);
    imwrite(I,['data/colorchecker-8bit/' dirOutput(i).name])
end

