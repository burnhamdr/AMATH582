clear all; close all; clc;
load Testdata
[r, c] = size(Undata);
L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks, ks, ks);
%% AVERAGE SIGNAL in 3D
Utsum = zeros(n,n,n);
for j=1:r
    Utsum = Utsum + fftn(reshape(Undata(j,:), n, n, n));
end
Utave = fftshift(Utsum)/r;
%isosurface of marble frequency signature after averaging
figure(5)
Utp = abs(Utave)./max(abs(Utave(:)));
isosurface(Kx,Ky,Kz,Utp,0.6)
title('Isosurface of Averaged Transformed Measurements')
xlabel('Wavenumber') 
ylabel('Wavenumber')
zlabel('Wavenumber')
axis([-8 8 -8 8 -8 8]), grid on, drawnow
%% Find center frequency
[M, I] = max(abs(Utp(:)));%find maximum amplitude value in average signal matrix
[row,col,vert] = ind2sub(size(Utp),I);
x_freq = Kx(row, col, vert); y_freq = Ky(row, col, vert); z_freq = Kz(row, col, vert);
%% Apply Filter
tau = 0.2;
filter = exp(-tau*((Kx - x_freq).^2 + (Ky - y_freq).^2 + (Kz - z_freq).^2));
xp = zeros(1,r); yp = zeros(1,r); zp = zeros(1,r);
for j=1:r
    Utr(:,:,:)=fftn(reshape(Undata(j,:), n, n, n));
    unft = filter.*fftshift(Utr);
    unus=ifftshift(unft);
    unf=ifftn(unus);
    [M, I] = max(abs(unf(:)));%find maximum value corresponding to marble position
    [row,col,vert] = ind2sub(size(unf), I);
    x_pos = X(row, col, vert); y_pos = Y(row, col, vert); z_pos = Z(row, col, vert);
    xp(j) = x_pos; yp(j) = y_pos; zp(j) = z_pos;
end
plot3(xp, yp, zp), grid on;
title('Marble Trajectory')
xlabel('Spatial Domain x') 
ylabel('Spatial Domain y')
zlabel('Spatial Domain z')
pos20 = [xp(20), yp(20), zp(20)]%final marble position