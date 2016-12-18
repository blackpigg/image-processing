list = dir('C:\Workspace\image processing\Dataset\train\net_7876\*.jpg');
fid = fopen('ImageList2.txt','w');
for i = 1 : length(list)
%     train_list_1(i,:) = list(i).name;
    fprintf(fid,'%s%s\n','net_7876\',list(i).name);
    
end
fclose(fid)