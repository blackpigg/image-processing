clc;clear all;close all
fid = fopen('bbox2.txt','r');
name = [];
coordinate = [];
cnt = 1;
temp = 1;
cropimgs = struct;
person_idx = 1;
while 1
    temp = fgetl(fid);
    if temp == -1
        break
    else
        x = findstr(temp,' ');
        
        if length(x) ~= 0
            img = imread(temp(1:x(1)-1));
            temp_coordinate = str2num(temp(x(1)+1:end));
            for j = 1 : length(temp_coordinate)/4
                coordinate = temp_coordinate(4*j-3:4*j);
                if 0 ~= coordinate(:)
                    cropimgs(cnt).img = img(coordinate(3):coordinate(4),coordinate(1):coordinate(2),:);
                else
                    coordinate(0==coordinate) = coordinate(0==coordinate)+1;
                    cropimgs(cnt).img = img(coordinate(3):coordinate(4),coordinate(1):coordinate(2),:);
                end
                cropimgs(cnt).name = temp(1:x(1)-1);
                cropimgs(cnt).rawimg = img;
                cropimgs(cnt).coordinate = coordinate;
                cropimgs(cnt).person_idx = person_idx;
%                 imshow(cropimgs(cnt).img); pause;                
                cnt = cnt + 1;    
            end
        end        
    end
    person_idx = person_idx + 1;
end
fclose(fid)