function output = full_evaluate()
    main_path = 'C:\Program Files (x99)\Others\download\nfa\train_0726_face_100_0.0001';
    all_catagory = [{'cat'} {'face'} {'bed'} {'duck'}];
    output = [];
    files = dir(main_path);
    files = files(3:length(files));
    for ii=1:length(files)
        if files(ii).isdir
            files(ii).result = evaluate(files(ii).name, 'face');
        end
    end
%     for ii=1:length(files)
%         if files(ii).isdir
%             for iii = 1:length(all_catagory)
%                 if strfind(files(ii).name, all_catagory{iii})
%                     files(ii).result = evaluate(files(ii).name, all_catagory{iii});
%                 end
%             end
%         end
%     end
    output = files;
end

function result = evaluate(model_path, catagory)
    main_path = 'C:\Program Files (x99)\Others\download\nfa\train_0726_face_100_0.0001\';
    all_catagory = [{'cat'} {'face'} {'bed'} {'duck'}];
    fprintf('Current evaluating: %s, in catagory: %s\n', model_path, catagory);
    result = [];
    try
        load([main_path model_path '\model']);  
        re_negative = zeros(1,30);
        for i = 1:length(all_catagory)
            fprintf('Current evaluate %s images\n', all_catagory{i});
            [temp_x, temp_y, im_train, im_test] = evaluation(config, net, all_catagory{i});
            if strcmp(catagory, all_catagory{i})
                re_train = temp_x;
                re_test = temp_y;
                imwrite(im_train, [main_path '\' sprintf('%s_train_%s.bmp', model_path, all_catagory{i})]);
                imwrite(im_test, [main_path '\' sprintf('%s_test_%s.bmp', model_path, all_catagory{i})]);
            else
                re_negative = re_negative + temp_x + temp_y;
                imwrite(im_train, [main_path '\' sprintf('%s_in_%s.bmp', model_path, all_catagory{i})]);
            end
        end
        re_negative = re_negative ./ 6;
        fig = plot(5:5:30, re_train(5:5:30), 5:5:30, re_test(5:5:30), 5:5:30, re_negative(5:5:30));
        legend(fig, 'Trainning Data', 'Testing Data', 'Negative Data');
        saveas(gcf, [main_path '\' model_path '_error.bmp']);
        result = [re_train(end) re_test(end) re_negative(end)];
    catch
        disp('Fail to find the model files');
        pause;
    end
end