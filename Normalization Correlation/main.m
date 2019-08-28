

img = imread('images/me.jfif');
img_face_clip = img(74:137, 186:243);

[indX indY] = find_template_2D(img_face_clip, img)

[y_s x_s] = size(img_face_clip);

imshow(img);
hold on;
rectangle('Position', [indY indX y_s x_s],'EdgeColor', 'b','LineWidth', 3 ); 