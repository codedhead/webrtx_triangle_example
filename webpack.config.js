const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');

const ROOT = path.resolve(__dirname);
const DESTINATION = path.resolve(__dirname, 'dist');

module.exports = (env) => {
  const isProd = env.production;
  const dev = !isProd;
  const mode = isProd ? 'production' : 'development';

  return {
    context: ROOT,
    mode,
    target: 'web',
    entry: './src/index.ts',
    output: {
      filename: 'index.js',
      path: DESTINATION,
    },
    module: {
      rules: [{
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      }, ],
    },
    resolve: {
      extensions: ['.ts', '.js'],
    },
    devtool: dev && 'inline-source-map',
    plugins: [
      new CopyWebpackPlugin({
        patterns: [{
          from: '*.wasm',
          context: path.resolve(__dirname, 'node_modules/webrtx/dist'),
        }],
      }),
    ],
  };
};