result= cell(0);
result2 = zeros(10,10000);
result3 = zeros(10,10000);

for i = 1:1000
    result{end+1} = squeeze(responseBackup{i}(14).x);
end

for i = 1:1000
    result2(:,(i-1)*100+1:100*i) = result{i};
end

shuffle2 = shuffle(90001:100000);
result3 = result2(:,90001:100000);