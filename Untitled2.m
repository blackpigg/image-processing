for i = 1 : 10000
    for j = 1: 10
        if 0 == mod(j,2)
            cropimgs(i).ftpoints(j) = cropimgs(i).ftpoints(j)-cropimgs(i).coordinate(3);
        else 
            cropimgs(i).ftpoints(j) = cropimgs(i).ftpoints(j)-cropimgs(i).coordinate(1);
        end
    end
end

for i = 1 : 10000
    [h1,w1,c1] = size(cropimgs(i).img);
    [h2,w2] = size(cropimgs(i).resizedimg);
    for j = 1: 10
        
        if 0 == mod(j,2)
            temp = cropimgs(i).ftpoints(j)/h1;
            cropimgs(i).ftpoints(j) = temp*h2;
        else 
            temp = cropimgs(i).ftpoints(j)/w1;
            cropimgs(i).ftpoints(j) = temp*w2;
        end
    end
end