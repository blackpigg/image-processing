result= cell(0);
for i = 1:5000
    result{end+1} = squeeze(responseBackup{i}(14).x);
end