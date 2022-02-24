
% UNDISTORT - Registration and gradient
reorientation

newgrad(i,:)=[grad_dirs(i,:) sqrt(sum(grad_dirs(i,:).^ 2))]*tform.T;
newgrad2(i,:)=grad_dirs(i,:)*tform.T(1: 3, 1: 3);
rDWI(:,:,:, i)=imwarp(data(:,:,:, i), tform, 'OutputView', imref3d(size(KK)));
end
for i=1:length(newgrad)
grad_dirs(i,:)=newgrad(i, 4). * newgrad(i, 1: 3)./ sqrt(sum(newgrad(i, 1: 3).^ 2));
end
grad_dirs = newgrad2;
grad_dirs(1: Param.nb0,:)=zeros(Param.nb0, 3);

for i = 1:length(bval)
rDWI(:,:,:, i)=smooth3(rDWI(:,:,:, i), 'gaussian', 5);
end