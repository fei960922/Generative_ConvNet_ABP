function full_evaluate(model_path, catagory)
    all_catagory = [{'cat'} {'face'} {'bed'} {'duck'}];
    printf('Current evaluating: %s, in catagory: %s', model_path, catagory);
    load(model_path);
    for i = 1:length(all_catagory)
        [x, y, im_output] = evaluation(config, net, all_catagory{i});
        if ~strcmp(catagory, all_catagory)
            re_train = temp_x;
            re_test = temp_y;
        else
            re_negative = temp_x + temp_y;
        end
    end
    re_negative = re_negative ./ 6;
    plot(5:5:30, re_train, 5:5:30, re_test, 5:5:30, re_negative);
    legend('Trainning Data', 'Testing Data', 'Negative Data');
end