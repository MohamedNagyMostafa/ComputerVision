

img = imread('images/me.jfif');
img_face_clip = img(74:137, 186:243);

[indX indY] = find_template_2D(img_face_clip, img)

[y_s x_s] = size(img_face_clip);
subplot(1,3,1);
imshow(img);
title('original');
subplot(1,3,2);
imshow(img_face_clip);
title('template');
subplot(1,3,3);
imshow(img);
title('After Norm. Corr.');
hold on;
rectangle('Position', [indY indX y_s x_s],'EdgeColor', 'b','LineWidth', 3 ); 