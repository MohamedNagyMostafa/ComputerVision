function [indX indY]= find_template_2D(temp,img)
    c = normxcorr2(temp,img);
    [indXP, indYP] = find(c == max(c(:)));
    
    indX = indXP - size(temp,2) + 1;
    indY = indYP - size(temp,2) + 1;
endfunction
%
%s = [-1 0 0 -1 1 0 1 0 0 1 1 -1;0 0 1 -1 0 1 1 1 0 0 -1 0;1 1 -1 0 1 -1 -1 -1 1 1 1 1;0 0 -1 -1 0 -1 -1 0 1 1 -1 1];
%t = [-1 1 ;-1 0];
%disp('signal'), disp([1:size(s,2);s]);
%disp('template'), disp([1:size(t,2);t]);

%[indX indY] = find_template_1D(t,s);
%disp('index:'),disp(indX),disp(' '),disp(indY);