import os
import datetime
import matplotlib.pyplot as plt
import csv

class Visualizer:
    def __init__(self, dir = "training_log"):
        self.dir = dir
        os.makedirs(dir, exist_ok= True)
        self.metrics = {
            'iteration': [],
            'supervised_loss': [],
            'reinforcement_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'eval_wins': [],
            'eval_losses': [],
            'game_length': [],
            'exploration_rate': [],
            'timestamp': []
        }

        self.csv = os.path.join(dir, 'metrics.csv')
        with open(self.csv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.metrics.keys())

        plt.ion()
        self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('Training progress', fontsize= 16)

        def update_plots(self):
            for ax in self.axs.flat:
                ax.clear()
            valid_indices = [i for i, loss in enumerate(self.metrics['reinforcement_loss']) if loss is not None]
            if valid_indices:
                iterations = [self.metrics['iteration'][i] for i in valid_indices]

                # Loss curves
                if any(loss is not None for loss in self.metrics['reinforcement_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['reinforcement_loss'][i] for i in valid_indices],'b-',
                                        label= 'Reinforcement Loss')
                if any(loss is not None for loss in self.metrics['supervised_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['supervised_loss'][i] for i in valid_indices],'b-',
                                        label= 'Supervised Loss')
                if any(loss is not None for loss in self.metrics['value_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['value_loss'][i] for i in valid_indices], '-g', label= 'Value loss')
                if any(loss is not None for loss in self.metrics['policy_loss']):
                    self.axs[0, 0].plot(iterations, [self.metrics['policy_loss'][i] for i in valid_indices], '-m', label= 'Policy Loss')
                self.axs[0, 0].set_title('Training loss')
                self.axz[0, 0].set_xlabel('Iterations')
                self.axz[0, 0].set_ylabel('Loss')
                self.axs[0, 0].legend()
                self.axs[0, 0].grid(True)

                # Evaluation results
                eval_indices = [i for i, wins in enumerate(self.metrics['eval_wins']) if wins is not None]
                if eval_indices:
                    eval_iterations = [self.metrics['iteration'][i] for i in eval_indices]
                    wins = [self.metrics['eval_wins'][i] for i in eval_indices]
                    losses = [self.metrics['eval_losses'][i] for i in eval_indices]
                    draws = [self.metrics['eval_losses'][i] for i in eval_indices]

                    self.axs[0, 1].plot(eval_iterations, wins, '-g', label= 'Wins')
                    self.axs[0, 1].plot(eval_iterations, draws, '-h', labels= 'Draws')
                    self.axs[0, 1].plot(eval_iterations, losses, '-r', labels= 'Losses')
                    self.axs[0, 1].set_title('Evaluation loss')
                    self.axs[0, 1].set_xlabel('Iterations')
                    self.axs[0, 1].set_ylabel('Games')
                    self.axs[0, 1].legend()
                    self.axs[0, 1].grid(True)

                    # Plot win rate
                    total_games = [w + d + l for w, d, l in zip(wins, draws, losses)]
                    win_rates = [w/t if t > 0 else 0 for w, t in zip(wins, total_games)]
                    self.axs[0, 2].plot(eval_iterations, win_rates, 'purple', label= 'Win Rate')
                    self.axs[0, 2].set_title('Win rate over time')
                    self.axs[0, 2].set_xlabel('Iteration')
                    self.axs[0, 2].set_ylabel('Win Rate')
                    self.axs[0, 2].set_ylim(0, 1)
                    self.axs[0, 2].legend()
                    self.axs[0, 2].grid(True)

                    # Elo rating estimate
                    elo_ratings = []
                    baseline_elo = 1200
                    for i in eval_indices:
                        wins = self.metrics['eval_wins'][i]
                        losses = self.metrics['eval_losses'][i]
                        draws = self.metrics['eval_draws'][i]
                        total_games = wins + losses + draws

                        if total_games > 0:
                            win_rate = wins / total_games
                            draw_rate = draws / total_games

                            expected_score = win_rate + 0.5 * draw_rate
                            elo_difference = 400 * math.log10(
                                expected_score / (1 - expected_score)) if expected_score < 1 else 400
                            elo_ratings.append(baseline_elo + elo_difference)
                        else:
                            elo_ratings.append(baseline_elo)
                    self.axs[1, 2].plot(eval_iterations, elo_ratings, 'red')
                    self.axs[1, 2].set_title('Estimated Elo ratings')
                    self.axs[1, 2].set_xlabel('Iteration')
                    self.axs[1, 2].set_ylabel('Elo')
                    self.axs[1, 2].grid(True)

                # Game length
                game_length = [i for i, length in enumerate(self.metrics['game_length']) if length is not None]
                if game_length:
                    game_iterations = [self.metrics['iteration'][i] for i in game_length]
                    game_lengths = [self.metrics['game_length'][i] for i in game_length]
                    self.axs[1, 0].plot(game_iterations, game_lengths, 'orange')
                    self.axs[1, 0].set_title('Average game length')
                    self.axs[1, 0].set_xlabel('Iterations')
                    self.axs[1, 0].set_ylabel('Moves')
                    self.axs[1, 0].grid(True)

                # Exploration rate
                exploration_indices = [i for i, rate in enumerate(self.metrics['exploration_rate']) if rate is not None]
                if exploration_indices:
                    exploration_iterations = [self.metrics['iteration'][i] for i in exploration_indices]
                    exploration_rates = [self.metrics['exploration_rate'][i] for i in exploration_indices]
                    self.axs[1, 1].plot(exploration_iterations, exploration_rates, 'brown')
                    self.axs[1, 1].set_title('Exploration rate')
                    self.axs[1, 1].set_xlabel('Iteration')
                    self.axs[1, 1].set_ylabel('Temperature')
                    self.axs[1, 1].grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        def save_plots(self):
            path = os.path.join(self.dir, f"training_plots_{datetime.now().strftime('%Y%m%d-%H%M')}.png")
            self.fig.savefig(path)
            print(f"Plots saved to {path}")

        def generate_report(self):
            path = os.path.join(self.dir, f"training_reports_{datetime.now().strftime('%Y%m%d-%H')}.txt")
            with open(path, 'w') as file:
                file.write("Training report \n")
                file.write("==============================\n\n")
                file.write(f"Generated on: {datetime.now().strftime('%Y%m%d-%H')}\n \n")
                file.write("Training statistics: \n")
                file.write(f"Total iterations: {len(self.metrics['iteration'])} \n")

                # Loss statistics
                valid_losses = [loss for loss in self.metrics['reinforcement_losses'] if loss is not None]
                if valid_losses:
                    file.write(f"Final reinforcement loss: {valid_losses[-1]:.4f} \n")
                    file.write(f"Minimum RL loss: {min(valid_losses):.4f} \n")

                # Evaluation statistics
                eval_wins = [w for w in self.metrics['eval_wins'] if w is not None]
                if eval_wins:
                    total_wins = sum(eval_wins)
                    total_draws = sum([d for d in self.metrics['eval_draws'] if d is not None])
                    total_losses = sum([l for l in self.metrics['eval_losses'] if l is not None])
                    total_games = total_wins + total_draws + total_losses

                    file.write(f"\nEvaluations results: \n")
                    file.write(f"Total games: {total_games} \n")
                    file.write(f"Wins: {total_wins} ({total_wins/total_games*100:.1f}%)\n")
                    file.write(f"Draws: {total_draws} ({total_draws/total_games*100:.1f}\n")
                    file.write(f"Losses: {total_losses} ({total_losses/total_games*100:.1f}\n")

                if len(self.metrics['timestamp']) > 1:
                    start_time = datetime.strptime(self.metrics['timestamp'][0], "%Y-%m-%d %H:%M:%S.%f")
                    end_time = datetime.strptime(self.metrics['timestamp'][-1], "%Y-%m-%d %H:%M:%S.%f")
                    duration = end_time - start_time

                    file.write(f"\n Training duration: {duration} \n")

            print(f"Report saved to {path}")


        def update_metrics(self, iteration, supervised_loss= None, reinforcement_loss= None,
                           value_loss= None, policy_loss= None, eval_results= None,
                           game_length= None, exploration_rate= None):
            self.metrics['iteration'].append(iteration)
            self.metrics['supervised_loss'].append(supervised_loss)
            self.metrics['reinforcement_loss'].append(reinforcement_loss)
            self.metrics['value_loss'].append(value_loss)
            self.metrics['policy_loss'].append(policy_loss)
            self.metrics['game_length'].append(game_length)
            self.metrics['exploration_rate'].append(exploration_rate)
            self.metrics['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            if eval_results:
                self.metrics['eval_wins'].append(eval_results['wins'])
                self.metrics['eval_losses'].append(eval_results['losses'])
                self.metrics['eval_draws'].append(eval_results['losses'])
            else:
                self.metrics['eval_wins'].append(None)
                self.metrics['eval_draws'].append(None)
                self.metrics['eval_losses'].append(None)

            with open(self.csv, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([iteration, supervised_loss, reinforcement_loss, value_loss, policy_loss,
                                    eval_results['wins'] if eval_results else None, eval_results['losses'] if eval_results else None,
                                    eval_results['draws'] if eval_results else None, game_length, exploration_rate,
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            self.update_plots()