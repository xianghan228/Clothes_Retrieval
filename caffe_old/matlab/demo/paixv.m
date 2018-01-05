function [ result ] = paixv( a )
%PAIXV Summary of this function goes here
%   Detailed explanation goes here
N = length(a);
for i = 1:N-1
    swap = 0;
    for j = 1:N-i
        if a(j)>a(j+1)
            temp = a(j);
            a(j) = a(j+1);
            a(j+1) = temp;
            swap = 1;
        end
    end
    if(~swap)
        break;
    end
    

end
result = a;
