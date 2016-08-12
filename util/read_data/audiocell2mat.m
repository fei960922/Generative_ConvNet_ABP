function audio_mat = audiocell2mat(audio_cell)
audio_mat = [];
if isempty(audio_cell)
    return;
end
numAudios = numel(audio_cell);
sz = size(audio_cell{1}');
if isa(audio_cell{1}, 'gpuArray')
    audio_mat = gpuArray(zeros([sz, numAudios], 'single'));
else
    audio_mat = zeros([sz, numAudios], 'single');
end
for i = 1:numAudios 
     % audio_mat : 1*60000*1*n_audio
      audio_mat(:,:,1,i) = audio_cell{i}';
 
end