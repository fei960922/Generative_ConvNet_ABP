function [audiodb, fn] = convert2imdb_audio(audio_mat)
% --------------------------------------------------------------------
numAudios = size(audio_mat, 4);
audiodb.audios.data = audio_mat ;
audiodb.audios.set = ones(1, numAudios);
audiodb.meta.sets = {'train', 'val', 'test'} ;

fn = @(audiodb,batch)getBatch(audiodb,batch);
end

function audio = getBatch(audiodb, batch)
% --------------------------------------------------------------------
audio = audiodb.audios.data(:,:,:,batch) ;
end